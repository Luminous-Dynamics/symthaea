<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * Trust Badge Component
   *
   * Displays PoGQ (Proof of Generalized Quality) trust scores with:
   * - Color-coded visual indicator
   * - Percentage display
   * - Optional detailed breakdown
   * - Hover tooltip
   * - Size variants (small, medium, large)
   * - Optional click to view full trust profile
   *
   * Trust Score Tiers:
   * - 90-100%: Exceptional (purple)
   * - 75-89%: Excellent (green)
   * - 60-74%: Good (blue)
   * - 40-59%: Fair (yellow)
   * - 0-39%: Poor (red)
   */

  import { createEventDispatcher } from 'svelte';

  // Props
  export let trustScore: number; // 0-100 or 0-1 (auto-converted)
  export let size: 'small' | 'medium' | 'large' = 'medium';
  export let showLabel: boolean = true;
  export let showIcon: boolean = true;
  export let clickable: boolean = false;
  export let agentId: string = ''; // For click navigation

  // Optional detailed breakdown
  export let breakdown: {
    transactionCount?: number;
    positiveReviews?: number;
    averageRating?: number;
    memberSince?: number;
  } | null = null;

  // State
  let showTooltip = false;

  const dispatch = createEventDispatcher<{
    click: { agentId: string; trustScore: number };
  }>();

  /**
   * Normalize trust score to 0-100 range
   */
  $: normalizedScore = trustScore > 1 ? trustScore : trustScore * 100;

  /**
   * Get trust tier based on score
   */
  $: trustTier = getTrustTier(normalizedScore);

  /**
   * Get display label
   */
  $: displayLabel = getTrustLabel(normalizedScore);

  /**
   * Get icon based on tier
   */
  $: trustIcon = getTrustIcon(trustTier);

  /**
   * Determine trust tier
   */
  function getTrustTier(score: number): string {
    if (score >= 90) return 'exceptional';
    if (score >= 75) return 'excellent';
    if (score >= 60) return 'good';
    if (score >= 40) return 'fair';
    return 'poor';
  }

  /**
   * Get trust label
   */
  function getTrustLabel(score: number): string {
    const tier = getTrustTier(score);
    const labels: Record<string, string> = {
      exceptional: 'Exceptional',
      excellent: 'Excellent',
      good: 'Good',
      fair: 'Fair',
      poor: 'Poor',
    };
    return labels[tier];
  }

  /**
   * Get trust icon
   */
  function getTrustIcon(tier: string): string {
    const icons: Record<string, string> = {
      exceptional: '👑',
      excellent: '⭐',
      good: '✓',
      fair: '○',
      poor: '⚠',
    };
    return icons[tier];
  }

  /**
   * Handle badge click
   */
  function handleClick() {
    if (clickable) {
      dispatch('click', { agentId, trustScore: normalizedScore });
    }
  }

  /**
   * Determine wrapper element based on interactivity
   */
  let badgeElement: 'div' | 'button' = 'div';
  $: badgeElement = clickable ? 'button' : 'div';

  /**
   * Format date to relative time
   */
  function formatMemberSince(timestamp: number): string {
    const now = Date.now();
    const diff = now - timestamp;
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    if (days < 30) return `${Math.floor(days / 7)} weeks`;
    if (days < 365) return `${Math.floor(days / 30)} months`;
    return `${Math.floor(days / 365)} years`;
  }
</script>

<svelte:element
  this={badgeElement}
  class="trust-badge"
  class:clickable={clickable}
  class:size-small={size === 'small'}
  class:size-medium={size === 'medium'}
  class:size-large={size === 'large'}
  class:tier-exceptional={trustTier === 'exceptional'}
  class:tier-excellent={trustTier === 'excellent'}
  class:tier-good={trustTier === 'good'}
  class:tier-fair={trustTier === 'fair'}
  class:tier-poor={trustTier === 'poor'}
  type={clickable ? 'button' : undefined}
  role={clickable ? 'button' : 'group'}
  aria-label={clickable ? `View trust profile for user with ${normalizedScore.toFixed(0)}% trust score` : undefined}
  on:click={clickable ? handleClick : undefined}
  on:mouseenter={() => (showTooltip = true)}
  on:mouseleave={() => (showTooltip = false)}
>
  <div class="badge-content">
    {#if showIcon}
      <span class="badge-icon">{trustIcon}</span>
    {/if}
    <span class="badge-score">{normalizedScore.toFixed(1)}%</span>
    {#if showLabel}
      <span class="badge-label">{displayLabel}</span>
    {/if}
  </div>

  {#if showTooltip && (breakdown || true)}
    <div class="badge-tooltip">
      <div class="tooltip-header">
        <span class="tooltip-icon">{trustIcon}</span>
        <span class="tooltip-title">PoGQ Trust Score</span>
      </div>

      <div class="tooltip-score">
        {normalizedScore.toFixed(1)}% · {displayLabel}
      </div>

      {#if breakdown}
        <div class="tooltip-breakdown">
          {#if breakdown.transactionCount !== undefined}
            <div class="breakdown-item">
              <span class="item-label">Transactions:</span>
              <span class="item-value">{breakdown.transactionCount}</span>
            </div>
          {/if}

          {#if breakdown.positiveReviews !== undefined}
            <div class="breakdown-item">
              <span class="item-label">Positive Reviews:</span>
              <span class="item-value">{breakdown.positiveReviews}</span>
            </div>
          {/if}

          {#if breakdown.averageRating !== undefined}
            <div class="breakdown-item">
              <span class="item-label">Average Rating:</span>
              <span class="item-value">{breakdown.averageRating.toFixed(1)}/5</span>
            </div>
          {/if}

          {#if breakdown.memberSince !== undefined}
            <div class="breakdown-item">
              <span class="item-label">Member Since:</span>
              <span class="item-value">{formatMemberSince(breakdown.memberSince)}</span>
            </div>
          {/if}
        </div>
      {:else}
        <div class="tooltip-info">
          Trust score based on transaction history, reviews, and community reputation.
        </div>
      {/if}

      {#if clickable}
        <div class="tooltip-action">Click to view full profile</div>
      {/if}
    </div>
  {/if}
</svelte:element>

<style>
  .trust-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-weight: 600;
    transition: all 0.2s;
    position: relative;
    user-select: none;
    border: none;
    background: transparent;
  }

  /* Size Variants */
  .size-small {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    gap: 0.25rem;
  }

  .size-medium {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    gap: 0.5rem;
  }

  .size-large {
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    gap: 0.75rem;
  }

  /* Clickable */
  .clickable {
    cursor: pointer;
    background: transparent;
  }

  .clickable:focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.8);
    outline-offset: 2px;
  }

  .clickable:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  /* Trust Tiers */
  .tier-exceptional {
    background: linear-gradient(135deg, #805ad5 0%, #6b46c1 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(128, 90, 213, 0.3);
  }

  .tier-excellent {
    background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(56, 161, 105, 0.3);
  }

  .tier-good {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(66, 153, 225, 0.3);
  }

  .tier-fair {
    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(237, 137, 54, 0.3);
  }

  .tier-poor {
    background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
    color: white;
    box-shadow: 0 2px 8px rgba(229, 62, 62, 0.3);
  }

  /* Badge Content */
  .badge-content {
    display: flex;
    align-items: center;
    gap: inherit;
  }

  .badge-icon {
    font-size: 1.25em;
  }

  .size-small .badge-icon {
    font-size: 1em;
  }

  .size-large .badge-icon {
    font-size: 1.5em;
  }

  .badge-score {
    font-weight: 700;
  }

  .badge-label {
    font-weight: 500;
    opacity: 0.9;
  }

  /* Tooltip */
  .badge-tooltip {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 50%;
    transform: translateX(-50%);
    background: white;
    border-radius: 0.5rem;
    padding: 1rem;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    min-width: 250px;
    z-index: 100;
    pointer-events: none;
  }

  .badge-tooltip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 8px solid transparent;
    border-top-color: white;
  }

  .tooltip-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
  }

  .tooltip-icon {
    font-size: 1.5rem;
  }

  .tooltip-title {
    font-weight: 600;
    color: #2d3748;
  }

  .tooltip-score {
    font-size: 1.25rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 1rem;
  }

  .tooltip-breakdown {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }

  .breakdown-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.875rem;
  }

  .item-label {
    color: #718096;
  }

  .item-value {
    font-weight: 600;
    color: #2d3748;
  }

  .tooltip-info {
    font-size: 0.75rem;
    color: #718096;
    line-height: 1.4;
    margin-bottom: 0.75rem;
  }

  .tooltip-action {
    font-size: 0.75rem;
    color: #4299e1;
    text-align: center;
    padding-top: 0.5rem;
    border-top: 1px solid #e2e8f0;
  }
</style>
