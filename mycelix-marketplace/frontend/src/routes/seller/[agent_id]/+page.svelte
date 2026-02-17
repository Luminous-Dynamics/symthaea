<script lang="ts">
  /**
   * Seller Profile Page
   *
   * Shows:
   * - Seller profile (from Holochain users zome)
   * - Marketplace stats (reviews)
   * - Knowledge-based author reputation (from Mycelix Knowledge)
   */

  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { initHolochainClient } from '$lib/holochain';
  import { getUserProfile, getReviewsForSeller } from '$lib/holochain/users';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import type { UserProfile, Review } from '$types';
  import { KnowledgeClient, type AuthorReputation } from '@mycelix/knowledge-client';

  $: sellerId = $page.params.agent_id;

  let loading = true;
  let error = '';
  let profile: UserProfile | null = null;
  let reviews: Review[] = [];

  let reputationLoading = false;
  let reputationError = '';
  let reputation: AuthorReputation | null = null;

  onMount(async () => {
    await loadSeller();
  });

  async function loadSeller() {
    loading = true;
    error = '';
    profile = null;
    reviews = [];
    reputation = null;

    if (!sellerId) {
      error = 'Invalid seller ID';
      loading = false;
      return;
    }

    try {
      const client = await initHolochainClient();
      profile = await getUserProfile(client, sellerId);
      reviews = await getReviewsForSeller(client, sellerId);

      void loadAuthorReputation(profile);
    } catch (e: any) {
      error = e?.message || 'Failed to load seller profile';
    } finally {
      loading = false;
    }
  }

  async function loadAuthorReputation(currentProfile: UserProfile | null) {
    if (!currentProfile) return;

    reputationLoading = true;
    reputationError = '';
    reputation = null;

    try {
      const appClient = await initHolochainClient();
      const knowledge = new KnowledgeClient(appClient as any, 'knowledge');

      // For now, use the agent_id as the author identifier.
      const authorDid = currentProfile.agent_id;
      reputation = await knowledge.inference.getAuthorReputation(authorDid);
    } catch (e: any) {
      reputationError = e?.message || 'Unable to load author reputation';
    } finally {
      reputationLoading = false;
    }
  }

  function formatTrustScore(score: number): string {
    const percentage = score > 1 ? score : score * 100;
    return `${percentage.toFixed(1)}%`;
  }
</script>

<div class="seller-profile">
  <div class="container">
    <ConnectionNotice />

    {#if loading}
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading seller profile...</p>
      </div>
    {:else if error || !profile}
      <div class="error-state">
        <span class="error-icon">⚠️</span>
        <h2>Seller Not Found</h2>
        <p>{error || 'This seller profile is unavailable.'}</p>
        <button class="btn btn-primary" on:click={() => goto('/browse')}>Back to marketplace</button>
      </div>
    {:else}
      <div class="header-row">
        <button class="back-link" on:click={() => goto('/browse')}>← Back to marketplace</button>
      </div>

      <div class="profile-card">
        <div class="avatar">
          {#if profile.avatar_cid}
            <img src={`https://ipfs.io/ipfs/${profile.avatar_cid}`} alt={profile.username} />
          {:else}
            <div class="avatar-placeholder">{profile.username.charAt(0).toUpperCase()}</div>
          {/if}
        </div>
        <div class="profile-main">
          <h1>{profile.username}</h1>
          <p class="member-since">
            Member since {new Date(profile.member_since).toLocaleDateString()}
          </p>
          <div class="profile-stats">
            <span class="pill">Marketplace trust: {formatTrustScore(profile.trust_score)}</span>
            <span class="pill">Sales: {profile.total_sales}</span>
            <span class="pill">Listings: {profile.total_listings}</span>
            <span class="pill">Average rating: {profile.average_rating.toFixed(1)}</span>
          </div>
          {#if profile.bio}
            <p class="bio">{profile.bio}</p>
          {/if}
        </div>
      </div>

      <div class="reputation-grid">
        <div class="knowledge-card">
          <div class="card-header">
            <p class="eyebrow">Knowledge reputation</p>
            <button
              class="refresh"
              on:click={() => loadAuthorReputation(profile)}
              disabled={reputationLoading}
            >
              {reputationLoading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
          {#if reputationLoading}
            <div class="loading-row">
              <div class="spinner small"></div>
              <p>Assessing epistemic track record…</p>
            </div>
          {:else if reputationError}
            <div class="error-chip">
              {reputationError}
            </div>
          {:else if reputation}
            <div class="metric-row">
              <span class="metric-label">Overall score</span>
              <span class="metric-value">{(reputation.overallScore * 100).toFixed(1)}%</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">Claims authored</span>
              <span class="metric-value">{reputation.claimsAuthored}</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">Verified true / false</span>
              <span class="metric-value">
                {reputation.claimsVerifiedTrue} / {reputation.claimsVerifiedFalse}
              </span>
            </div>
            <div class="metric-row">
              <span class="metric-label">Historical accuracy</span>
              <span class="metric-value">{(reputation.historicalAccuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">MATL trust</span>
              <span class="metric-value">{(reputation.matlTrust * 100).toFixed(1)}%</span>
            </div>
          {:else}
            <p class="subtle">No author reputation available yet.</p>
          {/if}
        </div>

        <div class="reviews-card">
          <div class="card-header">
            <p class="eyebrow">Reviews</p>
          </div>
          {#if reviews.length === 0}
            <p class="subtle">No reviews yet.</p>
          {:else}
            <div class="reviews-list">
              {#each reviews as review}
                <div class="review">
                  <div class="review-header">
                    <span class="rating">{'⭐'.repeat(review.rating)}</span>
                    <span class="date">
                      {new Date(review.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p class="comment">{review.comment}</p>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .seller-profile {
    min-height: 100vh;
    padding: 2rem 1rem;
    background: #f7fafc;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    gap: 1rem;
  }

  .spinner {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 3px solid #e2e8f0;
    border-top-color: #4299e1;
    animation: spin 1s linear infinite;
  }

  .spinner.small {
    width: 24px;
    height: 24px;
    border-width: 2px;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    background: white;
    border-radius: 0.5rem;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  }

  .error-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
  }

  .header-row {
    margin-bottom: 1rem;
  }

  .back-link {
    border: none;
    background: transparent;
    color: #4a5568;
    font-size: 0.9rem;
    cursor: pointer;
  }

  .profile-card {
    display: flex;
    gap: 1.5rem;
    padding: 1.5rem;
    background: white;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    margin-bottom: 1.5rem;
  }

  .avatar {
    width: 80px;
    height: 80px;
  }

  .avatar img {
    width: 100%;
    height: 100%;
    border-radius: 999px;
    object-fit: cover;
  }

  .avatar-placeholder {
    width: 100%;
    height: 100%;
    border-radius: 999px;
    background: #e2e8f0;
    display: grid;
    place-items: center;
    font-size: 1.8rem;
    font-weight: 700;
    color: #4a5568;
  }

  .profile-main h1 {
    margin: 0 0 0.25rem;
    font-size: 1.7rem;
    color: #1a202c;
  }

  .member-since {
    margin: 0 0 0.75rem;
    color: #718096;
    font-size: 0.9rem;
  }

  .profile-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }

  .pill {
    padding: 0.25rem 0.6rem;
    background: #edf2f7;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #4a5568;
  }

  .bio {
    margin: 0;
    color: #4a5568;
    font-size: 0.95rem;
  }

  .reputation-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
    gap: 1.25rem;
  }

  .knowledge-card,
  .reviews-card {
    background: white;
    border-radius: 0.5rem;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .eyebrow {
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366f1;
  }

  .refresh {
    border: 1px solid #d1d5db;
    background: #ffffff;
    border-radius: 999px;
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #374151;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .refresh:hover:not(:disabled) {
    border-color: #6366f1;
    color: #4338ca;
    box-shadow: 0 4px 10px rgba(99, 102, 241, 0.16);
  }

  .refresh:disabled {
    opacity: 0.65;
    cursor: default;
  }

  .loading-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #4b5563;
  }

  .error-chip {
    padding: 0.5rem 0.75rem;
    background: #fff5f5;
    color: #c53030;
    border: 1px solid #fed7d7;
    border-radius: 0.5rem;
    font-size: 0.85rem;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
  }

  .metric-label {
    color: #6b7280;
  }

  .metric-value {
    color: #111827;
    font-weight: 600;
  }

  .subtle {
    font-size: 0.9rem;
    color: #6b7280;
  }

  .reviews-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .review {
    padding: 0.75rem 0;
    border-bottom: 1px solid #e5e7eb;
  }

  .review:last-child {
    border-bottom: none;
  }

  .review-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
  }

  .rating {
    color: #fbbf24;
  }

  .date {
    font-size: 0.8rem;
    color: #9ca3af;
  }

  .comment {
    margin: 0;
    font-size: 0.9rem;
    color: #4b5563;
  }

  @media (max-width: 900px) {
    .reputation-grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>

