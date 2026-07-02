<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { initHolochainClient } from '$lib/holochain/client';
  import { createReview } from '$lib/holochain/users';
  import { notifications, isConnected } from '$lib/stores';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';

  // URL parameters
  let transactionHash = '';
  let listingHash = '';
  let listingTitle = '';
  let sellerName = '';

  // Form state
  let rating = 0;
  let hoveredRating = 0;
  let comment = '';

  // UI state
  let submitting = false;

  /**
   * Load URL parameters
   */
  onMount(() => {
    transactionHash = $page.url.searchParams.get('transaction') || '';
    listingHash = $page.url.searchParams.get('listing') || '';
    listingTitle = $page.url.searchParams.get('title') || 'Product';
    sellerName = $page.url.searchParams.get('seller') || 'Seller';

    if (!transactionHash || !listingHash) {
      notifications.error('Invalid Request', 'Missing transaction or listing information');
      setTimeout(() => {
        goto('/transactions');
      }, 2000);
    }
  });

  /**
   * Set rating
   */
  function setRating(value: number) {
    rating = value;
  }

  /**
   * Set hovered rating (for preview)
   */
  function setHoveredRating(value: number) {
    hoveredRating = value;
  }

  /**
   * Clear hovered rating
   */
  function clearHoveredRating() {
    hoveredRating = 0;
  }

  /**
   * Get star class
   */
  function getStarClass(starIndex: number): string {
    const effectiveRating = hoveredRating || rating;
    return starIndex <= effectiveRating ? 'star filled' : 'star';
  }

  /**
   * Validate form
   */
  function validateForm(): boolean {
    if (rating === 0) {
      notifications.error('Rating Required', 'Please select a star rating');
      return false;
    }

    if (!comment.trim()) {
      notifications.error('Comment Required', 'Please write a comment');
      return false;
    }

    if (comment.trim().length < 10) {
      notifications.error('Comment Too Short', 'Please write at least 10 characters');
      return false;
    }

    return true;
  }

  /**
   * Submit review
   */
  async function handleSubmit() {
    if (!validateForm()) return;

    submitting = true;

    try {
      const client = await initHolochainClient();

      await createReview(client, {
        listing_hash: listingHash,
        transaction_hash: transactionHash,
        rating,
        comment: comment.trim(),
      });

      notifications.success('Review Submitted', 'Your review has been published!');

      // Redirect to transactions page
      setTimeout(() => {
        goto('/transactions');
      }, 1500);
    } catch (e: any) {
      console.error('Failed to submit review:', e);
      notifications.error('Submission Failed', e.message || 'Failed to submit review');
    } finally {
      submitting = false;
    }
  }

  /**
   * Cancel and go back
   */
  function handleCancel() {
    goto('/transactions');
  }

  /**
   * Get rating description
   */
  function getRatingDescription(value: number): string {
    switch (value) {
      case 1:
        return 'Poor';
      case 2:
        return 'Fair';
      case 3:
        return 'Good';
      case 4:
        return 'Very Good';
      case 5:
        return 'Excellent';
      default:
        return 'Select a rating';
    }
  }
</script>

<main class="submit-review">
  <div class="container">
    <!-- Header -->
    <header class="page-header">
      <h1>Write a Review</h1>
      <p class="subtitle">Share your experience with other buyers</p>
    </header>

    <ConnectionNotice />

    <!-- Review Form -->
    <form on:submit|preventDefault={handleSubmit} class="review-form">
      <!-- Transaction Context -->
      <div class="context-card">
        <div class="context-label">Reviewing</div>
        <div class="context-title">{listingTitle}</div>
        <div class="context-seller">Sold by {sellerName}</div>
      </div>

      <!-- Star Rating -->
      <fieldset class="form-group">
        <legend>
          Your Rating <span class="required">*</span>
        </legend>

        <div class="star-rating" role="radiogroup" aria-label="Product rating">
          {#each [1, 2, 3, 4, 5] as starValue}
            <button
              type="button"
              class={getStarClass(starValue)}
              on:click={() => setRating(starValue)}
              on:mouseenter={() => setHoveredRating(starValue)}
              on:mouseleave={clearHoveredRating}
              aria-label="{starValue} stars"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="48"
                height="48"
                viewBox="0 0 24 24"
                fill="currentColor"
                stroke="none"
              >
                <path
                  d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
                />
              </svg>
            </button>
          {/each}
        </div>

        <p class="rating-description">{getRatingDescription(hoveredRating || rating)}</p>
      </fieldset>

      <!-- Comment -->
      <div class="form-group">
        <label for="comment-input">
          Your Review <span class="required">*</span>
        </label>
        <textarea
          id="comment-input"
          bind:value={comment}
          placeholder="Share your experience with this product and seller. What did you like? What could be improved?"
          rows="6"
          maxlength="1000"
          required
        />
        <small class="help-text">{comment.length}/1000 characters (minimum 10)</small>
      </div>

      <!-- Review Guidelines -->
      <div class="guidelines">
        <h3>Review Guidelines</h3>
        <ul>
          <li>Be honest and constructive</li>
          <li>Focus on your experience with the product and seller</li>
          <li>Avoid personal attacks or inappropriate language</li>
          <li>Reviews are public and cannot be edited after submission</li>
        </ul>
      </div>

      <!-- Form Actions -->
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" on:click={handleCancel} disabled={submitting}>
          Cancel
        </button>
        <button
          type="submit"
          class="btn btn-primary"
          disabled={submitting || !$isConnected}
          title={$isConnected ? '' : 'Connect to Holochain to submit your review'}
        >
          {#if submitting}
            Submitting Review...
          {:else}
            Submit Review
          {/if}
        </button>
      </div>
    </form>
  </div>
</main>

<style>
  .submit-review {
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1rem;
  }

  .container {
    max-width: 700px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    padding: 2rem;
  }

  .page-header {
    margin-bottom: 2rem;
    text-align: center;
  }

  .page-header h1 {
    font-size: 2rem;
    color: #2d3748;
    margin: 0 0 0.5rem 0;
  }

  .subtitle {
    color: #718096;
    font-size: 1rem;
    margin: 0;
  }

  .review-form {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  /* Context Card */
  .context-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 8px;
    color: white;
    text-align: center;
  }

  .context-label {
    font-size: 0.875rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
  }

  .context-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }

  .context-seller {
    font-size: 0.95rem;
    opacity: 0.9;
  }

  /* Form Group */
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  label {
    font-weight: 600;
    color: #2d3748;
    font-size: 1rem;
  }

  .required {
    color: #e53e3e;
  }

  /* Star Rating */
  .star-rating {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
  }

  .star {
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    transition: all 0.2s;
    color: #cbd5e0;
  }

  .star:hover {
    transform: scale(1.1);
  }

  .star.filled {
    color: #fbbf24;
  }

  .rating-description {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    color: #667eea;
    margin: 0.5rem 0 0 0;
    min-height: 1.5rem;
  }

  /* Textarea */
  textarea {
    padding: 1rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
    min-height: 150px;
    transition: border-color 0.2s;
  }

  textarea:focus {
    outline: none;
    border-color: #667eea;
  }

  .help-text {
    color: #718096;
    font-size: 0.875rem;
  }

  /* Guidelines */
  .guidelines {
    background: #f7fafc;
    border-left: 4px solid #667eea;
    padding: 1.5rem;
    border-radius: 8px;
  }

  .guidelines h3 {
    font-size: 1rem;
    color: #2d3748;
    margin: 0 0 1rem 0;
  }

  .guidelines ul {
    margin: 0;
    padding-left: 1.5rem;
    color: #4a5568;
  }

  .guidelines li {
    margin-bottom: 0.5rem;
  }

  .guidelines li:last-child {
    margin-bottom: 0;
  }

  /* Form Actions */
  .form-actions {
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
    padding-top: 1rem;
    border-top: 2px solid #e2e8f0;
  }

  .btn {
    padding: 0.875rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }

  .btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  }

  .btn-secondary {
    background: white;
    color: #2d3748;
    border: 2px solid #e2e8f0;
  }

  .btn-secondary:hover:not(:disabled) {
    background: #f7fafc;
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .container {
      padding: 1.5rem;
    }

    .star svg {
      width: 40px;
      height: 40px;
    }

    .form-actions {
      flex-direction: column;
    }

    .btn {
      width: 100%;
    }
  }
</style>
