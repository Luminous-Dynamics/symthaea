<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * Browse Marketplace
   *
   * Marketplace browsing interface with:
   * - Grid/list view of all listings
   * - Category filtering
   * - Price range filtering
   * - Search functionality
   * - Sort options
   */

  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { initHolochainClient } from '$lib/holochain';
  import { getAllListings } from '$lib/holochain/listings';
  import { notifications, isConnected, favorites, favoritesSet, proofRequests } from '$lib/stores';
  import { analyzeRisk } from '$lib/risk';
  import type { RiskSignal } from '$types';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import { getMockListings } from '$lib/mock/listings';
  import type { Listing, ListingCategory } from '$types';
  import { requestReputationProof } from '$lib/reputation';
  import RiskInsightDrawer from '$lib/components/RiskInsightDrawer.svelte';

  // Extended listing with trust score
  interface ListingWithTrust extends Listing {
    seller_trust_score?: number;
  }

  // State
  let loading = true;
  let error = '';
  let usingMockData = false;
  let allListings: ListingWithTrust[] = [];
  let filteredListings: ListingWithTrust[] = [];
  let lastUpdated = 0;
  $: lastUpdatedLabel = lastUpdated
    ? new Date(lastUpdated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';
  let proofLookup: Record<string, string> = {};
  let riskLookup: Record<string, RiskSignal> = {};
  let riskCache: Record<string, RiskSignal> = loadRiskCache();
  let activeFilters: { key: string; label: string }[] = [];

  // Filters
  let searchQuery = '';
  let selectedCategory: ListingCategory | 'All Categories' = 'All Categories';
  let minPrice = 0;
  let maxPrice = 10000;
  let sortBy: 'newest' | 'price-low' | 'price-high' | 'trust' = 'newest';
  let favoritesOnly = false;
  let activeTab: 'all' | 'favorites' = 'all';
  const gateways = ['https://ipfs.io/ipfs/', 'https://cloudflare-ipfs.com/ipfs/'];
  let hideHighRisk = true;
  let hideProofPending = true;
  let showProofFulfilledOnly = false;

  // View mode
  let viewMode: 'grid' | 'list' = 'grid';

  // Categories
  const categories: (ListingCategory | 'All Categories')[] = [
    'All Categories',
    'Electronics',
    'Fashion',
    'Home & Garden',
    'Sports & Outdoors',
    'Books & Media',
    'Toys & Games',
    'Health & Beauty',
    'Automotive',
    'Art & Collectibles',
    'Other',
  ];

  /**
   * Load all listings
   */
  onMount(async () => {
    // Restore favorites tab from query param
    const params = new URLSearchParams(window.location.search);
    const tab = params.get('tab');
    if (tab === 'favorites') {
      activeTab = 'favorites';
      favoritesOnly = true;
    }
    const hideRiskParam = params.get('hideRisk');
    const hideProofParam = params.get('hideProofPending');
    const proofFulfilledParam = params.get('proofFulfilled');
    hideHighRisk = hideRiskParam ? hideRiskParam === '1' : true;
    hideProofPending = hideProofParam ? hideProofParam === '1' : true;
    showProofFulfilledOnly = proofFulfilledParam === '1';

    await loadListings();
  });

  async function loadListings() {
    loading = true;
    error = '';

    try {
      const client = await initHolochainClient();
      const listings = await getAllListings(client);

      // Add placeholder trust scores (TODO: batch fetch seller profiles)
      allListings = listings.map((listing) => ({
        ...listing,
        seller_trust_score: 85, // Default value
      }));

      applyFilters();
      riskLookup = await analyzeRisk(allListings);

      notifications.success('Listings Loaded', `Found ${allListings.length} listings`);
      lastUpdated = Date.now();
      persistRiskCache(riskLookup);
    } catch (e: any) {
      // Fallback to mock data when offline or conductor unavailable
      const fallback = getMockListings();
      allListings = fallback.map((listing) => ({
        ...listing,
        seller_trust_score: 80,
      }));
      applyFilters();

      error = e.message || 'Failed to load listings; using offline preview data';
      notifications.warning('Using Offline Listings', error);
      usingMockData = true;
      riskLookup = await analyzeRisk(allListings);
      persistRiskCache(riskLookup);
      lastUpdated = Date.now();
    } finally {
      loading = false;
    }
  }

  /**
   * Update tab in URL without reloading
   */
  function updateTabParam(tab: 'all' | 'favorites') {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', tab);
    window.history.replaceState({}, '', url.toString());
  }

  function updateFiltersInUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', activeTab);
    url.searchParams.set('hideRisk', hideHighRisk ? '1' : '0');
    url.searchParams.set('hideProofPending', hideProofPending ? '1' : '0');
    url.searchParams.set('proofFulfilled', showProofFulfilledOnly ? '1' : '0');
    window.history.replaceState({}, '', url.toString());
  }

  function handleImageError(event: Event, cid?: string) {
    if (!cid) return;
    const img = event.currentTarget as HTMLImageElement;
    const current = Number(img.dataset.gw || '0');
    const next = current + 1;
    if (next < gateways.length) {
      img.dataset.gw = String(next);
      img.src = `${gateways[next]}${cid}`;
    } else {
      img.dataset.gw = String(next);
    }
  }

  async function requestProofInline(listing: ListingWithTrust) {
    try {
      await requestReputationProof(listing.seller_agent_id, listing.listing_hash || listing.id);
      notifications.success('Proof requested', 'Seller notified to publish proof.');
    } catch (e: any) {
      notifications.error(e?.message || 'Unable to request proof right now.');
    }
  }

  /**
   * Apply filters and sorting
   */
  function applyFilters() {
    let filtered = [...allListings];

    // Category filter
    if (selectedCategory !== 'All Categories') {
      filtered = filtered.filter((l) => l.category === selectedCategory);
    }

    // Price filter
    filtered = filtered.filter((l) => l.price >= minPrice && l.price <= maxPrice);

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (l) =>
          l.title.toLowerCase().includes(query) ||
          l.description.toLowerCase().includes(query) ||
          l.category.toLowerCase().includes(query)
      );
    }

    // Favorites filter
    if (favoritesOnly || activeTab === 'favorites') {
      const set = new Set($favoritesSet);
      filtered = filtered.filter((l) => set.has(l.listing_hash || l.id));
    }

    // Risk filter
    if (hideHighRisk) {
      filtered = filtered.filter((l) => {
        const score = riskLookup[l.listing_hash || l.id]?.score || 0;
        return score <= 0.5;
      });
    }

    if (hideProofPending) {
      filtered = filtered.filter(
        (l) => proofLookup[l.listing_hash || l.id] !== 'requested'
      );
    }

    if (showProofFulfilledOnly) {
      filtered = filtered.filter(
        (l) => proofLookup[l.listing_hash || l.id] === 'fulfilled'
      );
    }

    // Sort
    switch (sortBy) {
      case 'newest':
        filtered.sort((a, b) => b.created_at - a.created_at);
        break;
      case 'price-low':
        filtered.sort((a, b) => a.price - b.price);
        break;
      case 'price-high':
        filtered.sort((a, b) => b.price - a.price);
        break;
      case 'trust':
        filtered.sort((a, b) => (b.seller_trust_score || 0) - (a.seller_trust_score || 0));
        break;
    }

    filteredListings = filtered;
  }

  /**
   * Reactive filter application
   */
  $: if (allListings.length > 0) {
    applyFilters();
  }
  $: searchQuery, selectedCategory, minPrice, maxPrice, sortBy, hideHighRisk, hideProofPending, showProofFulfilledOnly, applyFilters(), updateFiltersInUrl();
  $: activeFilters = [
    hideHighRisk ? { key: 'hideRisk', label: 'Hide high-risk' } : null,
    hideProofPending ? { key: 'hideProof', label: 'Hide proof pending' } : null,
    showProofFulfilledOnly ? { key: 'proofFulfilled', label: 'Proof-fulfilled only' } : null,
  ].filter(Boolean) as { key: string; label: string }[];
  $: proofLookup = buildProofLookup();
  $: riskLookup = buildRiskLookup(riskLookup);

  /**
   * View listing details
   */
  function viewListing(listing_hash: string) {
    goto(`/listing/${listing_hash}`);
  }

  /**
   * Jump directly to trust section on listing detail
   */
  function viewTrustSection(listing_hash: string) {
    goto(`/listing/${listing_hash}#trust`);
  }

  function buildProofLookup() {
    const lookup: Record<string, string> = {};
    proofRequests.subscribe((requests) => {
      requests.forEach((req) => {
        if (req.listing_hash) {
          lookup[req.listing_hash] = req.status;
        }
      });
    })();
    return lookup;
  }

  function buildRiskLookup(current: Record<string, RiskSignal>) {
    const next = { ...riskCache, ...current };
    return next;
  }

  function persistRiskCache(risk: Record<string, RiskSignal>) {
    try {
      localStorage.setItem('mycelix_risk_cache', JSON.stringify(risk));
      riskCache = risk;
    } catch {
      // ignore
    }
  }

  function loadRiskCache(): Record<string, RiskSignal> {
    try {
      const raw = localStorage.getItem('mycelix_risk_cache');
      return raw ? (JSON.parse(raw) as Record<string, RiskSignal>) : {};
    } catch {
      return {};
    }
  }

  /**
   * Format date
   */
  function formatDate(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }
</script>

<div class="browse">
  <div class="container">
    <div class="browse-header">
      <div>
        <h1>Browse Marketplace</h1>
        <p class="subtitle">Discover unique items from trusted sellers</p>
        <div class="data-line">
          <span class={`data-pill ${usingMockData ? 'data-pill-mock' : 'data-pill-live'}`}>
            {usingMockData ? 'Preview data' : 'Live data'}
          </span>
          {#if lastUpdatedLabel}
            <span class="data-updated">Updated {lastUpdatedLabel}</span>
          {/if}
        </div>
      </div>
      <div class="header-actions">
        <div class="favorites-chip" title="Favorites saved on this device">
          ♥ {$favorites.length} favorites
        </div>
        <a class="favorites-link" href="/favorites">View favorites →</a>
        <button class="btn btn-tertiary" on:click={loadListings} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
    </div>

    <ConnectionNotice />
    {#if usingMockData}
      <div class="mock-badge">Offline preview data</div>
    {/if}

    <div class="tabs">
      <button class:active={activeTab === 'all'} on:click={() => { activeTab = 'all'; favoritesOnly = false; updateTabParam('all'); }}>
        All listings
      </button>
      <button class:active={activeTab === 'favorites'} on:click={() => { activeTab = 'favorites'; favoritesOnly = true; updateTabParam('favorites'); }}>
        Favorites ({$favorites.length})
      </button>
    </div>

    {#if loading}
      <!-- Loading State -->
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading listings...</p>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="error-state">
        <span class="error-icon">⚠️</span>
        <h2>Failed to Load Listings</h2>
        <p>{error}</p>
        <button class="btn btn-primary" on:click={loadListings}>Retry</button>
      </div>
    {:else}
      <!-- Filters and Controls -->
      <div class="controls-section">
        <div class="filters">
          <!-- Search -->
          <div class="filter-group">
            <input
              type="text"
              placeholder="Search listings..."
              bind:value={searchQuery}
              class="search-input"
            />
          </div>

          <!-- Category -->
          <div class="filter-group">
            <select bind:value={selectedCategory} class="filter-select">
              {#each categories as category}
                <option value={category}>{category}</option>
              {/each}
            </select>
          </div>

          <!-- Price Range -->
          <div class="filter-group price-filter">
            <label>
              <span>Min: ${minPrice}</span>
              <input
                type="range"
                min="0"
                max="10000"
                step="50"
                bind:value={minPrice}
                class="price-slider"
              />
            </label>
            <label>
              <span>Max: ${maxPrice}</span>
              <input
                type="range"
                min="0"
                max="10000"
                step="50"
                bind:value={maxPrice}
                class="price-slider"
              />
            </label>
          </div>

          <!-- Sort -->
          <div class="filter-group">
            <select bind:value={sortBy} class="filter-select">
              <option value="newest">Newest First</option>
              <option value="price-low">Price: Low to High</option>
              <option value="price-high">Price: High to Low</option>
              <option value="trust">Highest Trust Score</option>
            </select>
          </div>

          <div class="filter-group checkbox">
            <label>
              <input type="checkbox" bind:checked={hideHighRisk} />
              Hide high-risk
            </label>
          </div>

          <div class="filter-group checkbox">
            <label>
              <input type="checkbox" bind:checked={hideProofPending} />
              Hide proof pending
            </label>
          </div>
          <div class="filter-group checkbox">
            <label>
              <input type="checkbox" bind:checked={showProofFulfilledOnly} />
              Only proof-fulfilled
            </label>
          </div>
          <div class="filter-group checkbox">
            <label>
              <input type="checkbox" on:change={() => {
                hideProofPending = false;
                hideHighRisk = false;
                showProofFulfilledOnly = false;
                updateFiltersInUrl();
              }} />
              Clear filters
            </label>
          </div>

        </div>

        {#if hideHighRisk || hideProofPending}
          <div class="safe-mode-banner">
            <div>
              <strong>Safe mode on.</strong>
              <span>
                {hideHighRisk ? 'Hiding high-risk' : ''}{hideHighRisk && hideProofPending ? ' · ' : ''}{hideProofPending ? 'Hiding proof-pending' : ''}
              </span>
            </div>
            <button
              class="btn btn-tertiary"
              on:click={() => {
                hideHighRisk = false;
                hideProofPending = false;
              }}
            >
              Show all
            </button>
          </div>
        {/if}

        <!-- View Toggle -->
        <div class="view-toggle">
          <button
            class="view-btn"
            class:active={viewMode === 'grid'}
            on:click={() => (viewMode = 'grid')}
          >
            <span>⊞</span> Grid
          </button>
          <button
            class="view-btn"
            class:active={viewMode === 'list'}
            on:click={() => (viewMode = 'list')}
          >
            <span>☰</span> List
          </button>
        </div>
      </div>

      <!-- Results Count -->
      <div class="results-info">
        <p>{filteredListings.length} listings found</p>
        {#if activeFilters.length > 0}
          <div class="filter-chips">
            {#each activeFilters as filter}
              <button
                class="chip"
                on:click={() => {
                  if (filter.key === 'hideRisk') hideHighRisk = false;
                  if (filter.key === 'hideProof') hideProofPending = false;
                  if (filter.key === 'proofFulfilled') showProofFulfilledOnly = false;
                  updateFiltersInUrl();
                }}
              >
                {filter.label} ✕
              </button>
            {/each}
          </div>
        {:else}
          <button class="chip ghost" on:click={() => { showProofFulfilledOnly = true; updateFiltersInUrl(); }}>
            Proof-fulfilled only
          </button>
        {/if}
      </div>

      <!-- Listings Grid/List -->
      {#if filteredListings.length === 0}
        <div class="empty-state">
          <span>🔍</span>
          <h2>No listings found</h2>
          <p>Try adjusting your filters or search query</p>
          <button class="btn btn-secondary" on:click={() => {
            searchQuery = '';
            selectedCategory = 'All Categories';
            minPrice = 0;
            maxPrice = 10000;
          }}>
            Clear Filters
          </button>
        </div>
      {:else}
        {#if filteredListings.length === 0 && activeTab === 'favorites'}
          <div class="empty-state">
            <span>♥</span>
            <h2>No favorites yet</h2>
            <p>Tap the heart on any listing to add it here.</p>
            <button class="btn btn-secondary" on:click={() => { activeTab = 'all'; favoritesOnly = false; updateTabParam('all'); }}>
              Browse all listings
            </button>
          </div>
        {:else}
          <div class={`listings-container ${viewMode}`}>
            {#each filteredListings as listing}
              <button
                class="listing-card"
                on:click={() => viewListing(listing.listing_hash || listing.id)}
                on:keydown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    viewListing(listing.listing_hash || listing.id);
                  }
                }}
                aria-label="View listing for {listing.title}"
              >
                <div class="listing-image">
                  {#if listing.photos_ipfs_cids && listing.photos_ipfs_cids[0]}
                    <img
                      src="https://ipfs.io/ipfs/{listing.photos_ipfs_cids[0]}"
                      alt={listing.title}
                      data-gw="0"
                      on:error={(e) => handleImageError(e, listing.photos_ipfs_cids[0])}
                    />
                  {:else}
                    <div class="image-placeholder">📷</div>
                  {/if}
                  {#if usingMockData}
                    <div class="preview-badge">Preview</div>
                  {/if}
                  {#if listing.seller_trust_score}
                    <div class="trust-badge-wrapper">
                      <button
                        class="trust-badge"
                        type="button"
                        on:click|stopPropagation={() =>
                          viewTrustSection(listing.listing_hash || listing.id)}
                        title={usingMockData
                          ? 'Preview trust score · tap for full trust graph'
                          : proofLookup[listing.listing_hash || listing.id]
                            ? `Proof ${proofLookup[listing.listing_hash || listing.id]} · tap for full trust graph`
                            : 'Proof unknown · tap for full trust graph'}
                      >
                        {listing.seller_trust_score}% Trust
                        <span class="trust-preview" aria-label="Open trust graph">
                          {proofLookup[listing.listing_hash || listing.id]
                            ? `Proof ${proofLookup[listing.listing_hash || listing.id]}`
                            : 'Proof unknown'}
                        </span>
                        {#if usingMockData}
                          <span class="trust-preview secondary" title="Preview trust score">preview</span>
                        {:else if proofLookup[listing.listing_hash || listing.id] === 'requested'}
                          <span class="trust-preview pending" title="Proof requested">pending</span>
                        {/if}
                      </button>
                      <button
                        class="request-proof-btn"
                        type="button"
                        on:click|stopPropagation={() => requestProofInline(listing)}
                        title="Request proof from seller"
                      >
                        Request proof
                      </button>
                  </div>
                {/if}
                {#if riskLookup[listing.listing_hash || listing.id]?.score > 0.5}
                  <div
                    class="risk-chip"
                    title={riskLookup[listing.listing_hash || listing.id]?.flags?.[0] || 'Potential anomaly detected'}
                  >
                    Review suggested
                    <button
                      class="risk-action"
                      type="button"
                      on:click|stopPropagation={() => requestProofInline(listing)}
                      title="Request proof for this listing"
                    >
                      Request proof
                    </button>
                  </div>
                {/if}
                  <button
                    class={`favorite-btn ${$favoritesSet.has(listing.listing_hash || listing.id) ? 'active' : ''}`}
                    aria-label="Toggle favorite"
                    on:click|stopPropagation={() => favorites.toggle(listing.listing_hash || listing.id)}
                  >
                    {$favoritesSet.has(listing.listing_hash || listing.id) ? '♥' : '♡'}
                  </button>
                </div>

                <div class="listing-content">
                  <h3 class="listing-title">{listing.title}</h3>
                  <p class="listing-description">
                    {listing.description.slice(0, 100)}{listing.description.length > 100 ? '...' : ''}
                  </p>

                  <div class="listing-meta">
                    <span class="category-tag">{listing.category}</span>
                    <span class="date">{formatDate(listing.created_at)}</span>
                  </div>

                  <div class="listing-footer">
                    <span class="price">${listing.price.toFixed(2)}</span>
                    {#if listing.quantity_available}
                      <span class="quantity">{listing.quantity_available} available</span>
                    {/if}
                  </div>
                  {#if riskLookup[listing.listing_hash || listing.id]}
                    <div class="risk-insight">
                      <RiskInsightDrawer
                        risk={riskLookup[listing.listing_hash || listing.id]}
                        proof={proofLookup[listing.listing_hash || listing.id] || null}
                        open={false}
                      />
                    </div>
                  {/if}
                </div>
              </button>
            {/each}
          </div>
        {/if}
      {/if}
    {/if}
  </div>
</div>

<style>
  .browse {
    min-height: 100vh;
    padding: 2rem 1rem;
    background: #f7fafc;
  }

  .tabs {
    display: inline-flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 0.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  }

  .tabs button {
    border: none;
    background: transparent;
    padding: 0.5rem 0.9rem;
    border-radius: 0.6rem;
    cursor: pointer;
    font-weight: 600;
    color: #4a5568;
  }

  .tabs button.active {
    background: #edf2f7;
    color: #2d3748;
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

  .favorites-chip {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    color: #e53e3e;
    font-weight: 700;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    height: fit-content;
  }

  .header-actions {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    flex-wrap: wrap;
  }

  .btn-tertiary {
    background: #edf2f7;
    color: #1a202c;
    border: 1px solid #cbd5e0;
  }

  .btn-tertiary:hover:not(:disabled) {
    background: #e2e8f0;
  }

  .favorites-link {
    color: #4299e1;
    font-weight: 700;
    text-decoration: none;
  }

  .favorites-link:hover {
    text-decoration: underline;
  }

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Header */
  .browse-header {
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
  }

  .browse-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .subtitle {
    font-size: 1.125rem;
    color: #718096;
  }

  .data-line {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
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

  /* Controls Section */
  .controls-section {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 2rem;
    background: white;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    flex-wrap: wrap;
  }

  .filters {
    display: flex;
    gap: 1rem;
    flex: 1;
    flex-wrap: wrap;
  }

  .safe-mode-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fffaf0;
    border: 1px solid #f6ad55;
    color: #9a3412;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    gap: 0.75rem;
  }

.filter-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

  .filter-group.checkbox {
    flex-direction: row;
    align-items: center;
    gap: 0.35rem;
  }

  .filter-group.checkbox input {
    width: 16px;
    height: 16px;
  }

  .search-input {
    padding: 0.75rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    font-size: 1rem;
    min-width: 250px;
  }

  .search-input:focus {
    outline: none;
    border-color: #4299e1;
  }

  .filter-select {
    padding: 0.75rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    font-size: 1rem;
    background: white;
    min-width: 180px;
  }

  .price-filter {
    min-width: 200px;
  }

  .price-filter label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .price-filter span {
    font-size: 0.875rem;
    color: #4a5568;
  }

  .price-slider {
    width: 100%;
  }

  /* View Toggle */
  .view-toggle {
    display: flex;
    gap: 0.5rem;
  }

  .view-btn {
    padding: 0.75rem 1rem;
    border: 1px solid #e2e8f0;
    background: white;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .view-btn:hover {
    border-color: #4299e1;
  }

  .view-btn.active {
    background: #4299e1;
    border-color: #4299e1;
    color: white;
  }

  /* Results Info */
  .results-info {
    margin-bottom: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .results-info p {
    color: #718096;
    font-size: 0.875rem;
  }

  .filter-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .chip {
    background: #edf2f7;
    color: #1a202c;
    border: 1px solid #cbd5e0;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-weight: 700;
    cursor: pointer;
    font-size: 0.85rem;
  }

  .chip.ghost {
    background: white;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    background: white;
    border-radius: 0.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .empty-state span {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .empty-state h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .empty-state p {
    color: #718096;
    margin-bottom: 2rem;
  }

  /* Listings Container */
  .listings-container.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.5rem;
  }

  .listings-container.list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  /* Listing Card */
  .listing-card {
    background: white;
    border-radius: 0.5rem;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: all 0.3s;
  }

  .listing-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .listings-container.list .listing-card {
    display: flex;
    flex-direction: row;
  }

  .listing-image {
    position: relative;
    width: 100%;
    height: 200px;
    overflow: hidden;
  }

  .listings-container.list .listing-image {
    width: 250px;
    height: auto;
    flex-shrink: 0;
  }

  .listing-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .image-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f7fafc;
    font-size: 3rem;
  }

  .trust-badge {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    padding: 0.25rem 0.75rem;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 0.4rem;
    font-size: 0.75rem;
    font-weight: 700;
    color: #38a169;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border: 1px solid #c6f6d5;
    cursor: pointer;
  }

  .trust-preview {
    background: rgba(56, 161, 105, 0.12);
    color: #2f855a;
    padding: 0.1rem 0.4rem;
    border-radius: 999px;
    font-size: 0.65rem;
    text-transform: uppercase;
  }

  .trust-preview.secondary {
    background: rgba(221, 107, 32, 0.12);
    color: #c05621;
  }

  .trust-preview.pending {
    background: rgba(59, 130, 246, 0.12);
    color: #2563eb;
  }

  .risk-chip {
    margin-top: 0.35rem;
    background: #fff7ed;
    color: #c2410c;
    border: 1px solid #fed7aa;
    border-radius: 0.5rem;
    padding: 0.4rem 0.6rem;
    font-size: 0.8rem;
    font-weight: 700;
    display: inline-flex;
    gap: 0.15rem;
  }

  .risk-action {
    background: transparent;
    border: none;
    color: #b45309;
    font-weight: 700;
    margin-left: 0.35rem;
    cursor: pointer;
    text-decoration: underline;
  }

  .favorite-btn {
    position: absolute;
    top: 0.75rem;
    left: 0.75rem;
    border: none;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 999px;
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    font-size: 1rem;
    cursor: pointer;
    transition: transform 0.1s ease, box-shadow 0.1s ease, color 0.1s ease;
    color: #4a5568;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
  }

  .preview-badge {
    position: absolute;
    top: 0.75rem;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255, 255, 255, 0.92);
    color: #dd6b20;
    border: 1px solid #fbd38d;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-weight: 700;
    font-size: 0.75rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  }

  .favorite-btn.active {
    color: #e53e3e;
  }

  .favorite-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.16);
  }

  /* Listing Content */
  .listing-content {
    padding: 1.25rem;
  }

  .listing-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.5rem;
    display: -webkit-box;
    line-clamp: 2;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .listing-description {
    font-size: 0.875rem;
    color: #718096;
    margin-bottom: 1rem;
    line-height: 1.5;
  }

  .listing-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .category-tag {
    padding: 0.25rem 0.75rem;
    background: #edf2f7;
    color: #4a5568;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .date {
    font-size: 0.75rem;
    color: #a0aec0;
  }

  .listing-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .price {
    font-size: 1.5rem;
    font-weight: 700;
    color: #38a169;
  }

  .quantity {
    font-size: 0.875rem;
    color: #718096;
  }

  .risk-insight {
    margin-top: 0.5rem;
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
    .controls-section {
      flex-direction: column;
    }

    .filters {
      flex-direction: column;
      width: 100%;
    }

    .filter-group,
    .search-input,
    .filter-select {
      width: 100%;
      min-width: 0;
    }

    .listings-container.list .listing-card {
      flex-direction: column;
    }

    .listings-container.list .listing-image {
      width: 100%;
      height: 200px;
    }
  }
</style>
