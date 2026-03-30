<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * User Dashboard
   *
   * Central hub showing:
   * - User profile with trust score
   * - Activity statistics
   * - Recent transactions
   * - Active listings
   * - Quick actions
   */

  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { initHolochainClient } from '$lib/holochain';
  import { getMyProfile } from '$lib/holochain/users';
  import { getMyListings } from '$lib/holochain/listings';
  import { getMyPurchases, getMySales } from '$lib/holochain/transactions';
  import { notifications, isConnected } from '$lib/stores';
  import { proofRequests } from '$lib/stores';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import { getMockProfile } from '$lib/mock/profile';
  import { getMockTransactions } from '$lib/mock/transactions';
  import { getMockListings } from '$lib/mock/listings';
  import type { UserProfile, Listing, Transaction, ProofRequest } from '$types';

  // State
  let loading = true;
  let error = '';
  let profile: UserProfile | null = null;
  let activeListings: Listing[] = [];
  let recentTransactions: Transaction[] = [];
  let usingMockData = false;
  let lastUpdated = 0;
  let recentProofs: ProofRequest[] = [];

  $: lastUpdatedLabel = lastUpdated
    ? new Date(lastUpdated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';
  $: proofRequests.subscribe((requests) => {
    recentProofs = requests.slice(0, 5);
  });

  /**
   * Load dashboard data
   */
  onMount(() => {
    loadDashboard();
  });

  async function loadDashboard() {
    loading = true;
    error = '';
    usingMockData = false;

    try {
      const client = await initHolochainClient();

      // Fetch all data in parallel
      const [userProfile, listings, purchases, sales] = await Promise.all([
        getMyProfile(client),
        getMyListings(client),
        getMyPurchases(client),
        getMySales(client),
      ]);

      profile = userProfile;
      activeListings = listings.filter((l) => l.status === 'active').slice(0, 5);

      // Combine and sort transactions
      const allTransactions = [...purchases, ...sales].sort(
        (a, b) => b.created_at - a.created_at
      );
      recentTransactions = allTransactions.slice(0, 5);
      lastUpdated = Date.now();

      notifications.success('Dashboard Loaded', `Welcome back, ${profile.username}!`);
    } catch (e: any) {
      const mockProfile = getMockProfile();
      const mockTx = getMockTransactions();
      profile = mockProfile;
      activeListings = getMockListings().filter((l) => l.status === 'active').slice(0, 5);
      recentTransactions = mockTx.slice(0, 5);
      usingMockData = true;
      error = e.message || 'Using offline preview data';
      notifications.warning('Offline Preview', error);
      lastUpdated = Date.now();
    } finally {
      loading = false;
    }
  }

  /**
   * Format date to relative time
   */
  function formatDate(timestamp: number): string {
    const now = Date.now();
    const diff = now - timestamp;
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
    if (days < 365) return `${Math.floor(days / 30)} months ago`;
    return `${Math.floor(days / 365)} years ago`;
  }

  /**
   * Format trust score
   */
  function formatTrustScore(score: number): string {
    // Handle both 0-1 and 0-100 formats
    const percentage = score > 1 ? score : score * 100;
    return `${percentage.toFixed(1)}%`;
  }
</script>

<div class="dashboard">
  <div class="container">
    {#if loading}
      <!-- Loading State -->
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="error-state">
        <span class="error-icon">⚠️</span>
        <h2>Failed to Load Dashboard</h2>
        <p>{error}</p>
        <button class="btn btn-primary" on:click={() => window.location.reload()}>
          Retry
        </button>
      </div>
    {:else if profile}
      <ConnectionNotice />
      {#if usingMockData}
        <div class="mock-badge">Offline preview data</div>
      {/if}
      <!-- Dashboard Content -->
      <div class="dashboard-header">
        <div class="profile-section">
          <div class="avatar">
            {#if profile.avatar_cid}
              <img
                src="https://ipfs.io/ipfs/{profile.avatar_cid}"
                alt="{profile.username} avatar"
              />
            {:else}
              <div class="avatar-placeholder">
                {profile.username.charAt(0).toUpperCase()}
              </div>
            {/if}
          </div>
          <div class="profile-info">
            <h1>{profile.username}</h1>
            {#if usingMockData}
              <span class="preview-pill">Preview</span>
            {/if}
            <p class="member-since">Member since {formatDate(profile.member_since)}</p>
            <div class="data-line">
              <span class={`data-pill ${usingMockData ? 'data-pill-mock' : 'data-pill-live'}`}>
                {usingMockData ? 'Preview data' : 'Live data'}
              </span>
              {#if lastUpdatedLabel}
                <span class="data-updated">Updated {lastUpdatedLabel}</span>
              {/if}
            </div>
            <div class="trust-score">
              <span class="trust-label">Trust Score:</span>
              <span class="trust-value">{formatTrustScore(profile.trust_score)}</span>
              {#if profile.is_verified}
                <span class="verified-badge">✓ Verified</span>
              {/if}
            </div>
          </div>
        </div>

        <div class="quick-actions">
          <button class="btn btn-tertiary" on:click={loadDashboard} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh data'}
          </button>
          <button class="btn btn-primary" on:click={() => goto('/create-listing')} disabled={!$isConnected}>
            + Create Listing
          </button>
          <button class="btn btn-secondary" on:click={() => goto('/browse')}>
            Browse Marketplace
          </button>
        </div>
      </div>

      <!-- Stats Grid -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon">📦</div>
          <div class="stat-value">{profile.total_listings}</div>
          <div class="stat-label">Total Listings</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">💰</div>
          <div class="stat-value">{profile.total_sales}</div>
          <div class="stat-label">Sales Completed</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">🛒</div>
          <div class="stat-value">{profile.total_purchases}</div>
          <div class="stat-label">Purchases Made</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">⭐</div>
          <div class="stat-value">{profile.average_rating.toFixed(1)}</div>
          <div class="stat-label">Avg Rating ({profile.total_reviews} reviews)</div>
        </div>
      </div>

      <!-- Main Content Grid -->
      <div class="content-grid">
        <!-- Recent Transactions -->
        <div class="card">
          <div class="card-header">
            <h2>Recent Transactions</h2>
            <button class="link-button" on:click={() => goto('/transactions')}>
              View All →
            </button>
          </div>
          {#if recentTransactions.length === 0}
            <div class="empty-state">
              <span>📭</span>
              <p>No transactions yet</p>
              <button class="btn btn-secondary" on:click={() => goto('/browse')}>
                Start Shopping
              </button>
            </div>
          {:else}
            <div class="transaction-list">
              {#each recentTransactions as tx}
                <button
                  class="transaction-item"
                  on:click={() => goto(`/transactions#${tx.id}`)}
                  on:keydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      goto(`/transactions#${tx.id}`);
                    }
                  }}
                  aria-label="View transaction {tx.id.slice(0, 8)}"
                >
                  <div class="transaction-info">
                    <span class="transaction-title">Transaction #{tx.id.slice(0, 8)}...</span>
                    <span class="transaction-date">{formatDate(tx.created_at)}</span>
                  </div>
                  <div class="transaction-meta">
                    <span class={`status-badge status-${tx.status}`}>{tx.status}</span>
                    <span class="transaction-amount">${tx.total_price.toFixed(2)}</span>
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Active Listings -->
        <div class="card">
          <div class="card-header">
            <h2>Active Listings</h2>
            <button class="link-button" on:click={() => goto('/my-listings')}>
              Manage All →
            </button>
          </div>
          {#if activeListings.length === 0}
            <div class="empty-state">
              <span>📦</span>
              <p>No active listings</p>
              <button class="btn btn-primary" on:click={() => goto('/create-listing')}>
                Create Listing
              </button>
            </div>
          {:else}
            <div class="listing-list">
              {#each activeListings as listing}
                <button
                  class="listing-item"
                  on:click={() => goto(`/listing/${listing.listing_hash}`)}
                  on:keydown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      goto(`/listing/${listing.listing_hash}`);
                    }
                  }}
                  aria-label="View listing {listing.title}"
                >
                  <div class="listing-image">
                    {#if listing.photos_ipfs_cids && listing.photos_ipfs_cids[0]}
                      <img
                        src="https://ipfs.io/ipfs/{listing.photos_ipfs_cids[0]}"
                        alt={listing.title}
                      />
                    {:else}
                      <div class="listing-placeholder">📷</div>
                    {/if}
                  </div>
                  <div class="listing-info">
                    <h3>{listing.title}</h3>
                    <p class="listing-price">${listing.price.toFixed(2)}</p>
                    <p class="listing-meta">
                      {listing.quantity_available} available · {formatDate(listing.created_at)}
                    </p>
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Proofs & Trust Inbox -->
        <div class="card">
          <div class="card-header">
            <h2>Proofs & Trust Inbox</h2>
            <button class="link-button" on:click={() => goto('/transactions')}>
              View Transactions →
            </button>
          </div>
          {#if recentProofs.length === 0}
            <div class="empty-state small">
              <span>🔍</span>
              <p>No proof activity yet</p>
            </div>
          {:else}
            <ul class="proof-list">
              {#each recentProofs as proof}
                <li class="proof-item">
                  <div>
                    <p class="proof-target">
                      {proof.listing_hash ? `Listing ${proof.listing_hash.slice(0, 8)}...` : 'Seller proof'}
                    </p>
                    <p class="proof-meta">
                      {proof.agent_id.slice(0, 8)} · {new Date(proof.updated_at).toLocaleTimeString()}
                    </p>
                  </div>
                  <span class={`proof-pill ${proof.status}`}>
                    {proof.status}
                  </span>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .dashboard {
    min-height: 100vh;
    padding: 2rem 1rem;
    background: #f7fafc;
  }

  .mock-badge {
    margin-bottom: 1rem;
    padding: 0.75rem 1rem;
    background: #fffaf0;
    border: 1px solid #f6ad55;
    color: #7b341e;
    border-radius: 0.5rem;
    font-weight: 600;
  }

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Loading State */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    gap: 1rem;
  }

  .spinner {
    width: 50px;
    height: 50px;
    border: 4px solid #e2e8f0;
    border-top-color: #4299e1;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Error State */
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    background: white;
    border-radius: 0.5rem;
    padding: 3rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .error-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .error-state h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .error-state p {
    color: #718096;
    margin-bottom: 2rem;
  }

  /* Dashboard Header */
  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    gap: 2rem;
    flex-wrap: wrap;
  }

  .profile-section {
    display: flex;
    gap: 1.5rem;
    align-items: center;
  }

  .avatar {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    overflow: hidden;
  }

  .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .avatar-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 2rem;
    font-weight: 700;
  }

  .profile-info h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 0.25rem;
  }

  .member-since {
    color: #718096;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
  }

  .data-line {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .data-pill {
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid transparent;
  }

  .data-pill-live {
    background: #dcfce7;
    color: #166534;
    border-color: #bbf7d0;
  }

  .data-pill-mock {
    background: #fff7ed;
    color: #c2410c;
    border-color: #fed7aa;
  }

  .data-updated {
    color: #4a5568;
    font-size: 0.9rem;
  }

  .trust-score {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .trust-label {
    color: #4a5568;
    font-size: 0.875rem;
  }

  .preview-pill {
    display: inline-block;
    margin-left: 0.5rem;
    background: #fffaf0;
    color: #dd6b20;
    border: 1px solid #f6ad55;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
  }

  .trust-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38a169;
  }

  .verified-badge {
    padding: 0.25rem 0.5rem;
    background: #c6f6d5;
    color: #22543d;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .quick-actions {
    display: flex;
    gap: 1rem;
  }

  .btn-tertiary {
    background: #edf2f7;
    color: #1a202c;
    border: 1px solid #cbd5e0;
  }

  .btn-tertiary:hover:not(:disabled) {
    background: #e2e8f0;
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  .stat-card {
    background: white;
    border-radius: 0.5rem;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    text-align: center;
  }

  .stat-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 0.25rem;
  }

  .stat-label {
    color: #718096;
    font-size: 0.875rem;
  }

  /* Content Grid */
  .content-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
  }

  .card {
    background: white;
    border-radius: 0.5rem;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .card-header h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #2d3748;
  }

  .link-button {
    background: none;
    border: none;
    color: #4299e1;
    font-size: 0.875rem;
    cursor: pointer;
    transition: color 0.2s;
  }

  .link-button:hover {
    color: #2b6cb0;
  }

  /* Proofs widget */
  .proof-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .proof-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.75rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    background: #f8fafc;
  }

  .proof-target {
    margin: 0;
    font-weight: 700;
    color: #1f2937;
  }

  .proof-meta {
    margin: 0.15rem 0 0;
    color: #6b7280;
    font-size: 0.9rem;
  }

  .proof-pill {
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    text-transform: capitalize;
    border: 1px solid transparent;
  }

  .proof-pill.pending {
    background: #eef2ff;
    color: #4338ca;
    border-color: #c7d2fe;
  }

  .proof-pill.requested {
    background: #dbeafe;
    color: #1d4ed8;
    border-color: #bfdbfe;
  }

  .proof-pill.fulfilled {
    background: #dcfce7;
    color: #166534;
    border-color: #bbf7d0;
  }

  .proof-pill.denied {
    background: #fee2e2;
    color: #b91c1c;
    border-color: #fecaca;
  }

  .empty-state.small {
    padding: 1.5rem 1rem;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    text-align: center;
  }

  .empty-state span {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .empty-state p {
    color: #718096;
    margin-bottom: 1.5rem;
  }

  /* Transaction List */
  .transaction-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .transaction-item {
    padding: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .transaction-item:hover {
    border-color: #4299e1;
    box-shadow: 0 2px 8px rgba(66, 153, 225, 0.1);
  }

  .transaction-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .transaction-title {
    font-weight: 600;
    color: #2d3748;
  }

  .transaction-date {
    font-size: 0.875rem;
    color: #a0aec0;
  }

  .transaction-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .status-pending {
    background: #feebc8;
    color: #7c2d12;
  }

  .status-shipped {
    background: #bee3f8;
    color: #2c5282;
  }

  .status-delivered,
  .status-completed {
    background: #c6f6d5;
    color: #22543d;
  }

  .transaction-amount {
    font-weight: 700;
    color: #2d3748;
  }

  /* Listing List */
  .listing-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .listing-item {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .listing-item:hover {
    border-color: #4299e1;
    box-shadow: 0 2px 8px rgba(66, 153, 225, 0.1);
  }

  .listing-image {
    width: 80px;
    height: 80px;
    border-radius: 0.375rem;
    overflow: hidden;
    flex-shrink: 0;
  }

  .listing-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .listing-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f7fafc;
    font-size: 2rem;
  }

  .listing-info h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.25rem;
  }

  .listing-price {
    font-size: 1.125rem;
    font-weight: 700;
    color: #38a169;
    margin-bottom: 0.25rem;
  }

  .listing-meta {
    font-size: 0.875rem;
    color: #718096;
  }

  /* Buttons */
  .btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 0.375rem;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
  }

  .btn-primary {
    background: #4299e1;
    color: white;
  }

  .btn-primary:hover {
    background: #3182ce;
  }

  .btn-secondary {
    background: #e2e8f0;
    color: #2d3748;
  }

  .btn-secondary:hover {
    background: #cbd5e0;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .content-grid {
      grid-template-columns: 1fr;
    }

    .dashboard-header {
      flex-direction: column;
      align-items: flex-start;
    }

    .quick-actions {
      width: 100%;
      flex-direction: column;
    }

    .quick-actions .btn {
      width: 100%;
    }
  }
</style>
