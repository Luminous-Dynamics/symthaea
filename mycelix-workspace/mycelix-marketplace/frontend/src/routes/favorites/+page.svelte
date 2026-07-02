<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import { favorites, favoritesSet, notifications } from '$lib/stores';
  import { initHolochainClient } from '$lib/holochain';
  import { getListing } from '$lib/holochain/listings';
  import { getMockListingWithContext } from '$lib/mock/listings';
  import type { ListingWithContext } from '$types';

  let loading = true;
  let error = '';
  let listingContexts: ListingWithContext[] = [];
  let usingMockData = false;
  const gateways = ['https://ipfs.io/ipfs/', 'https://cloudflare-ipfs.com/ipfs/'];

  onMount(async () => {
    await loadFavorites();
  });

  async function loadFavorites() {
    loading = true;
    error = '';
    listingContexts = [];
    usingMockData = false;

    const ids = $favoritesSet;

    if (ids.size === 0) {
      loading = false;
      return;
    }

    try {
      const client = await initHolochainClient();
      const contexts: ListingWithContext[] = [];

      for (const id of ids) {
        const ctx = await getListing(client, id);
        contexts.push(ctx);
      }

      listingContexts = contexts;
      notifications.success('Favorites loaded', `${listingContexts.length} saved items`);
    } catch (e: any) {
      // fallback to mock data
      const contexts: ListingWithContext[] = [];
      ids.forEach((id) => {
        const mock = getMockListingWithContext(id);
        if (mock) contexts.push(mock);
      });
      listingContexts = contexts;
      usingMockData = true;
      error = e.message || 'Using offline preview data';
      notifications.warning('Offline preview', error);
    } finally {
      loading = false;
    }
  }

  function viewListing(id: string) {
    goto(`/listing/${id}`);
  }

  function clearAll() {
    const confirmClear = window.confirm('Remove all favorites?');
    if (confirmClear) {
      favorites.clear();
      listingContexts = [];
    }
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
</script>

<main class="favorites-page">
  <div class="container">
    <header class="page-header">
      <div>
        <p class="eyebrow">Mycelix</p>
        <h1>Favorites</h1>
        <p class="subtitle">Saved listings on this device</p>
      </div>
      <div class="header-actions">
        <div class="fav-count">♥ {$favorites.length}</div>
        {#if $favorites.length > 0}
          <button class="btn btn-secondary" on:click={clearAll}>Clear all</button>
        {/if}
      </div>
    </header>

    <ConnectionNotice />
    {#if usingMockData}
      <div class="mock-badge">Offline preview data</div>
    {/if}

    {#if loading}
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading favorites...</p>
      </div>
    {:else if $favorites.length === 0}
      <div class="empty-state">
        <span>♡</span>
        <h2>No favorites yet</h2>
        <p>Tap the heart on any listing to save it here.</p>
        <button class="btn btn-primary" on:click={() => goto('/browse')}>Browse listings</button>
      </div>
    {:else if listingContexts.length === 0}
      <div class="empty-state">
        <span>⚠️</span>
        <h2>Favorites unavailable</h2>
        <p>We could not load details. Try reconnecting or refresh.</p>
        <button class="btn btn-secondary" on:click={loadFavorites}>Retry</button>
      </div>
    {:else}
      <div class="listings-container">
        {#each listingContexts as ctx}
          <button class="listing-card" on:click={() => viewListing(ctx.listing.listing_hash || ctx.listing.id)}>
            <div class="listing-image">
              {#if ctx.listing.photos_ipfs_cids && ctx.listing.photos_ipfs_cids[0]}
                <img
                  src={`https://ipfs.io/ipfs/${ctx.listing.photos_ipfs_cids[0]}`}
                  alt={ctx.listing.title}
                  data-gw="0"
                  on:error={(e) => handleImageError(e, ctx.listing.photos_ipfs_cids[0])}
                />
              {:else}
                <div class="image-placeholder">📷</div>
              {/if}
              {#if usingMockData}
                <div class="preview-badge">Preview</div>
              {/if}
            </div>
            <div class="listing-content">
              <h3>{ctx.listing.title}</h3>
              <p class="price">${ctx.listing.price.toFixed(2)}</p>
              <p class="category">{ctx.listing.category}</p>
              <p class="seller">{ctx.seller.username}</p>
            </div>
          </button>
        {/each}
      </div>
    {/if}
  </div>
</main>

<style>
  .favorites-page {
    min-height: 100vh;
    background: #f7fafc;
    padding: 2rem 1rem;
  }

  .container {
    max-width: 1100px;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.75rem;
    color: #718096;
    margin: 0;
  }

  h1 {
    margin: 0.25rem 0;
    font-size: 2rem;
    color: #2d3748;
  }

  .subtitle {
    margin: 0;
    color: #718096;
  }

  .fav-count {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 0.5rem 0.9rem;
    color: #e53e3e;
    font-weight: 700;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
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

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    gap: 0.75rem;
    background: white;
    border-radius: 0.75rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
  }

  .empty-state span {
    font-size: 3rem;
  }

  .listings-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
  }

  .listing-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    overflow: hidden;
    text-align: left;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }

  .listing-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
  }

  .listing-image {
    width: 100%;
    height: 180px;
    background: #f7fafc;
    position: relative;
  }

  .listing-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .image-placeholder {
    width: 100%;
    height: 100%;
    display: grid;
    place-items: center;
    color: #a0aec0;
    font-size: 2rem;
  }

  .preview-badge {
    position: absolute;
    top: 0.5rem;
    left: 0.5rem;
    background: rgba(255, 255, 255, 0.95);
    color: #dd6b20;
    border: 1px solid #fbd38d;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-weight: 700;
    font-size: 0.75rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  }

  .listing-content {
    padding: 1rem;
    display: grid;
    gap: 0.35rem;
  }

  .listing-content h3 {
    margin: 0;
    font-size: 1.1rem;
    color: #2d3748;
  }

  .price {
    font-weight: 700;
    color: #38a169;
  }

  .category,
  .seller {
    color: #718096;
    font-size: 0.9rem;
    margin: 0;
  }

  .btn {
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    font-weight: 700;
    cursor: pointer;
  }

  .btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }

  .btn-secondary {
    background: white;
    color: #2d3748;
    border: 1px solid #e2e8f0;
  }

  .spinner {
    width: 48px;
    height: 48px;
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
</style>
