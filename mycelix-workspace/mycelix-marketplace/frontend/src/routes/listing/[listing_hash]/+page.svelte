<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * Listing Detail Page
   *
   * Shows complete product information:
   * - Product photos (IPFS gallery)
   * - Title, description, price, quantity
   * - Seller profile and trust score
   * - Product reviews
   * - Add to cart / Buy now actions
   */

  import { onMount, tick } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { initHolochainClient } from '$lib/holochain';
  import { getListing } from '$lib/holochain/listings';
  import { cartItems } from '$lib/stores/cart';
  import { notifications, isConnected, favorites, favoritesSet } from '$lib/stores';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import TrustGraph from '$lib/components/TrustGraph.svelte';
  import ProofTrail from '$lib/components/ProofTrail.svelte';
  import ThermodynamicHUD from '$lib/components/ThermodynamicHUD.svelte';
  import { loadReputationBundle, requestReputationProof } from '$lib/reputation';
  import { getProofStatus, markProofFulfilled } from '$lib/stores';
  import RiskChip from '$lib/components/RiskChip.svelte';
  import RiskInsightDrawer from '$lib/components/RiskInsightDrawer.svelte';
  import { analyzeRisk } from '$lib/risk';
  import { getMockListingWithContext } from '$lib/mock/listings';
  import { initKnowledgeClient, type ListingKnowledgeSnapshot } from '$lib/knowledge/listing-knowledge';
  import { toDiscreteEpistemic } from '@mycelix/knowledge-client';
  import { notifications } from '$lib/stores';
  import type {
    ListingWithContext,
    ProofTrailItem,
    Review,
    SellerInfo,
    TrustGraphSnapshot,
  } from '$types';

  // Route parameter
  $: listing_hash = $page.params.listing_hash;

  // State
  let loading = true;
  let error = '';
  let usingMockData = false;
  let listing: ListingWithContext['listing'] | null = null;
  let seller: SellerInfo | null = null;
  let reviews: Review[] = [];
  let trustGraph: TrustGraphSnapshot | null = null;
  let proofTrail: ProofTrailItem[] = [];
  let trustLoading = false;
  let trustUsingMock = false;
  let trustError = '';
  let trustUpdatedAt = 0;
  let trustSectionEl: HTMLElement | null = null;
  let trustHighlight = false;
  let proofRequesting = false;
  let proofRequestMessage = '';
  let proofRequestStatus: 'idle' | 'pending' | 'requested' | 'fulfilled' | 'denied' = 'idle';
  let proofUpdateMessage = '';
  let previousProofCount = 0;
  let listingPhotos: string[] = [];
  let listingId = '';
  let listingTitle = '';
  let selectedPhotoCid: string | undefined;
  let listingRisk: import('$types').RiskSignal | null = null;
  let showRiskModal = false;
  let knowledgeSnapshot: ListingKnowledgeSnapshot | null = null;
  let knowledgeLoading = false;
  let knowledgeError = '';
  let knowledgeEpistemic: ReturnType<typeof toDiscreteEpistemic> | null = null;
  let verificationMarketLoading = false;

  // Thermodynamic state
  $: totalEntropy = (knowledgeSnapshot?.thermodynamic?.production_joules ?? 0) + 
                    (knowledgeSnapshot?.thermodynamic?.logistics_joules ?? 0);
  $: isEntropyVerified = knowledgeSnapshot?.thermodynamic?.verified ?? false;

  // Purchase state
  let quantity = 1;
  let purchasing = false;
  let selectedImageIndex = 0;
  const gateways = ['https://ipfs.io/ipfs/', 'https://cloudflare-ipfs.com/ipfs/'];

  $: listingPhotos = listing?.photos_ipfs_cids ?? [];
  $: listingId = listing?.listing_hash || listing?.id || '';
  $: listingTitle = listing?.title ?? 'Listing';
  $: selectedPhotoCid = listingPhotos[selectedImageIndex];
  $: if (selectedImageIndex >= listingPhotos.length) {
    selectedImageIndex = 0;
  }
  $: if (seller && listing_hash) {
    const status = getProofStatus(seller.agent_id, listing_hash);
    if (status) proofRequestStatus = status;
  }

  /**
   * Load listing data
   */
  onMount(async () => {
    await loadListing();
  });

  async function loadListing() {
    loading = true;
    error = '';

    // Validate listing_hash parameter
    if (!listing_hash) {
      error = 'Invalid listing ID';
      notifications.error('Invalid Listing', 'Listing ID is missing');
      loading = false;
      return;
    }

    try {
      const client = await initHolochainClient();
      const listingData = await getListing(client, listing_hash);

      listing = listingData.listing;
      seller = listingData.seller;
      reviews = listingData.reviews || [];
      const riskScores = await analyzeRisk([listingData.listing]);
      listingRisk = riskScores[listing_hash] || null;
      if (seller?.agent_id) {
        void loadReputation(seller.agent_id);
      }
      void loadKnowledgeSnapshot(listingData.listing);

      notifications.success('Listing Loaded', listing.title);
    } catch (e: any) {
      const fallback = getMockListingWithContext(listing_hash);
      if (fallback) {
        listing = fallback.listing;
        seller = fallback.seller;
        reviews = fallback.reviews || [];
        error = e.message || 'Using offline listing preview';
        notifications.warning('Offline Preview', error);
        usingMockData = true;
        const riskScores = await analyzeRisk([fallback.listing]);
        listingRisk = riskScores[listing_hash] || null;
        if (seller?.agent_id) {
          void loadReputation(seller.agent_id);
        }
      } else {
        error = e.message || 'Failed to load listing';
        notifications.error('Loading Failed', error);
      }
    } finally {
      loading = false;
    }
  }

  /**
   * Load verifiable reputation surface for seller
   */
  async function loadReputation(agentId: string, force: boolean = false) {
    trustLoading = true;
    trustError = '';
    trustUsingMock = false;
    trustGraph = null;
    proofTrail = [];
    trustUpdatedAt = 0;

    try {
      const bundle = await loadReputationBundle(agentId, { forceRefresh: force });
      trustGraph = bundle.graph;
      proofTrail = bundle.proofTrail;
      trustUsingMock = bundle.usingMock;
      if (bundle.error) {
        trustError = bundle.error;
      }
      trustUpdatedAt = Date.now();
      if (seller?.agent_id && listing_hash && (proofTrail?.length || 0) > 0) {
        markProofFulfilled(seller.agent_id, listing_hash);
        proofRequestStatus = 'fulfilled';
        if ((proofTrail?.length || 0) > previousProofCount) {
          proofUpdateMessage = 'New proof added to this listing.';
          previousProofCount = proofTrail.length;
          setTimeout(() => (proofUpdateMessage = ''), 4000);
        }
      }
    } catch (e: any) {
      trustError = e?.message || 'Unable to load trust graph';
    } finally {
      trustLoading = false;
    }

    await maybeScrollToTrust();
  }

  async function maybeScrollToTrust() {
    if ($page.url.hash !== '#trust') return;
    await tick();
    if (trustSectionEl) {
      trustSectionEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      trustHighlight = true;
      setTimeout(() => {
        trustHighlight = false;
      }, 1500);
    }
  }

  /**
   * Simulate proof request to seller (placeholder until zome wired)
   */
  async function requestProof() {
    if (!seller) return;
    proofRequesting = true;
    proofRequestMessage = '';
    proofRequestStatus = 'pending';
    try {
      const res = await requestReputationProof(seller.agent_id, listing_hash);
      proofRequestMessage =
        res.message ||
        `Proof request sent to ${seller.username}. You will see new attestations when published.`;
      if (res.usingMock && 'error' in res) {
        proofRequestMessage += ' (preview mode)';
      }
      proofRequestStatus = 'requested';
    } catch (e: any) {
      proofRequestMessage = e?.message || 'Unable to request proof right now.';
      proofRequestStatus = 'idle';
    } finally {
      proofRequesting = false;
    }
  }

  function messageSeller() {
    showRiskModal = false;
    notifications.info('Message seller', 'Use proof request or seller profile contact to reach the seller.');
  }

  /**
   * Add to cart
   */
  function addToCart(showNotification: boolean = true) {
    if (!listing || !seller || !listing_hash) return;

    cartItems.addItem({
      listing_hash,
      title: listing.title,
      price: listing.price,
      quantity,
      photo_cid: listing.photos_ipfs_cids[0],
      seller_agent_id: seller.agent_id,
      seller_name: seller.username,
    });

    if (showNotification) {
      notifications.success('Added to Cart', `${quantity}x ${listing.title}`);
    }
  }

  /**
   * Buy now (direct purchase)
   */
  function buyNow() {
    if (!listing || !seller || !listing_hash) return;

    purchasing = true;
    addToCart(false);
    notifications.success('Ready to checkout', `Review shipping and payment for ${quantity}x ${listing.title}`);
    goto('/checkout');
  }

  /**
   * Format date
   */
  function formatDate(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
  }

  /**
   * Load knowledge snapshot for this listing from Mycelix Knowledge (best-effort).
   */
  async function loadKnowledgeSnapshot(listingData: ListingWithContext['listing']) {
    knowledgeLoading = true;
    knowledgeError = '';
    knowledgeSnapshot = null;
    try {
      const client = await initKnowledgeClient();
      knowledgeSnapshot = await client.getListingKnowledgeSnapshot(listingData);
      knowledgeEpistemic =
        knowledgeSnapshot && knowledgeSnapshot.claim
          ? toDiscreteEpistemic(knowledgeSnapshot.claim.classification)
          : null;
    } catch (e: any) {
      knowledgeError = e?.message || 'Knowledge graph context unavailable';
    } finally {
      knowledgeLoading = false;
    }
  }

  /**
   * Request a verification market for the current listing's claim.
   */
  async function requestVerificationMarket() {
    if (!knowledgeSnapshot || !knowledgeSnapshot.claim || !knowledgeSnapshot.verificationRecommendation) return;

    verificationMarketLoading = true;
    try {
      const client = await initKnowledgeClient();
      const closesAt = Date.now() + 7 * 24 * 60 * 60 * 1000; // 7 days from now
      const marketHash = await client.requestVerificationMarketForClaim(
        knowledgeSnapshot.claim.id,
        knowledgeSnapshot.verificationRecommendation.suggestedTargetE,
        0.7,
        closesAt,
        ['marketplace', 'listing']
      );

      notifications.success(
        'Verification market requested',
        marketHash ? `Market reference: ${String(marketHash).slice(0, 12)}…` : 'Request submitted.'
      );
    } catch (e: any) {
      notifications.error('Verification request failed', e?.message || 'Unable to request verification market.');
    } finally {
      verificationMarketLoading = false;
    }
  }

  /**
   * Format trust score
   */
  function formatTrustScore(score: number): string {
    const percentage = score > 1 ? score : score * 100;
    return `${percentage.toFixed(1)}%`;
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

<div class="listing-detail">
  <div class="container">
    <ConnectionNotice />
    {#if usingMockData}
      <div class="mock-badge">Offline preview data</div>
    {/if}

    {#if loading}
      <!-- Loading State -->
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading listing...</p>
      </div>
    {:else if error || !listing}
      <!-- Error State -->
      <div class="error-state">
        <span class="error-icon">⚠️</span>
        <h2>Listing Not Found</h2>
        <p>{error || 'This listing does not exist or has been removed'}</p>
        <button class="btn btn-primary" on:click={() => goto('/browse')}>
          Browse Marketplace
        </button>
      </div>
    {:else}
      <!-- Breadcrumb -->
      <div class="breadcrumb">
        <button on:click={() => goto('/browse')}>Browse</button>
        <span>›</span>
        <button on:click={() => goto(`/browse?category=${listing?.category}`)}>
          {listing?.category}
        </button>
        <span>›</span>
        <span>{listing?.title}</span>
      </div>

      <!-- Main Content -->
      <div class="listing-grid">
        <!-- Left: Photos -->
        <div class="photos-section">
          <div class="main-photo">
            {#if usingMockData}
              <div class="preview-badge">Preview</div>
            {/if}
            {#if selectedPhotoCid}
              <img
                src="https://ipfs.io/ipfs/{selectedPhotoCid}"
                alt={listingTitle}
                data-gw="0"
                on:error={(e) => handleImageError(e, selectedPhotoCid)}
              />
            {:else}
              <div class="photo-placeholder">📷</div>
            {/if}
          </div>

          {#if listingPhotos.length > 1}
            <div class="photo-thumbnails">
              {#each listingPhotos as cid, index}
                <button
                  class="thumbnail"
                  class:active={index === selectedImageIndex}
                  on:click={() => (selectedImageIndex = index)}
                >
                  <img
                    src="https://ipfs.io/ipfs/{cid}"
                    alt="Thumbnail {index + 1}"
                    data-gw="0"
                    on:error={(e) => handleImageError(e, cid)}
                  />
                </button>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Right: Details -->
        <div class="details-section">
          <div class="title-row">
            <h1>{listing.title}</h1>
            <button
              class={`favorite-toggle ${listingId && $favoritesSet.has(listingId) ? 'active' : ''}`}
              on:click={() => listingId && favorites.toggle(listingId)}
              aria-label="Toggle favorite"
            >
              {listingId && $favoritesSet.has(listingId) ? '♥' : '♡'}
            </button>
          </div>

          <div class="price-section">
            <span class="price">${listing.price.toFixed(2)}</span>
            {#if listing.quantity_available}
              <span class="availability">
                {listing.quantity_available} available
              </span>
            {/if}
          </div>

          <div class="category-tag">{listing.category}</div>
          
          <ThermodynamicHUD 
            total_joules={totalEntropy} 
            production_joules={knowledgeSnapshot?.thermodynamic?.production_joules ?? 0}
            logistics_joules={knowledgeSnapshot?.thermodynamic?.logistics_joules ?? 0}
            verified={isEntropyVerified}
          />

          <div class="risk-row">
            <RiskChip risk={listingRisk} expanded={true} />
            {#if listingRisk && listingRisk.score > 0.5}
              <button class="btn btn-tertiary" on:click={() => (showRiskModal = true)}>
                Why this risk?
              </button>
            {/if}
          </div>

          <!-- Seller Info -->
          {#if seller}
            <div class="seller-card">
              <div class="seller-avatar">
                {#if seller.avatar_cid}
                  <img src="https://ipfs.io/ipfs/{seller.avatar_cid}" alt={seller.username} />
                {:else}
                  <div class="avatar-placeholder">
                    {seller.username.charAt(0).toUpperCase()}
                  </div>
                {/if}
              </div>
              <div class="seller-info">
                <button class="seller-name" on:click={() => goto(`/seller/${seller?.agent_id}`)}>
                  {seller?.username}
                </button>
                <div class="seller-stats">
                  <span class="trust-score">{formatTrustScore(seller.trust_score)} Trust</span>
                  <span class="rating">
                    ⭐ {seller.average_rating?.toFixed(1) || 'N/A'} ({seller.total_sales} sales)
                  </span>
                </div>
                <p class="member-since">Member since {formatDate(seller.member_since)}</p>
              </div>
            </div>
          {/if}

          <!-- Knowledge Graph Snapshot -->
          <div class="knowledge-section">
            <div class="knowledge-header">
              <p class="eyebrow">Knowledge graph</p>
              {#if listing}
                <button
                  class="knowledge-refresh"
                  on:click={() => loadKnowledgeSnapshot(listing)}
                  disabled={knowledgeLoading}
                >
                  {knowledgeLoading ? 'Refreshing…' : 'Refresh'}
                </button>
              {/if}
            </div>
            {#if knowledgeLoading}
              <div class="knowledge-loading">
                <div class="spinner small"></div>
                <p>Loading epistemic context…</p>
              </div>
            {:else if knowledgeError}
              <div class="knowledge-error">
                Epistemic context unavailable: {knowledgeError}
              </div>
            {:else if knowledgeSnapshot && knowledgeSnapshot.claim && knowledgeSnapshot.credibility}
              <div class="knowledge-card">
                <p class="knowledge-title">
                  Claim: <span>{knowledgeSnapshot.claim.content}</span>
                </p>
                <p class="knowledge-metric">
                  Credibility: <strong>{(knowledgeSnapshot.credibility.overallScore * 100).toFixed(1)}%</strong>
                </p>
                {#if knowledgeEpistemic}
                  <div class="knowledge-epistemic">
                    <div class="axis-row">
                      <span class="axis-label">Empirical (E):</span>
                      <span class="axis-value">
                        {(knowledgeSnapshot.claim.classification.empirical * 100).toFixed(0)}% ·
                        {knowledgeEpistemic.empirical}
                      </span>
                    </div>
                    <div class="axis-row">
                      <span class="axis-label">Normative (N):</span>
                      <span class="axis-value">
                        {(knowledgeSnapshot.claim.classification.normative * 100).toFixed(0)}% ·
                        {knowledgeEpistemic.normative}
                      </span>
                    </div>
                    <div class="axis-row">
                      <span class="axis-label">Mythic (M):</span>
                      <span class="axis-value">
                        {(knowledgeSnapshot.claim.classification.mythic * 100).toFixed(0)}% ·
                        {knowledgeEpistemic.mythic}
                      </span>
                    </div>
                  </div>
                {/if}
                {#if knowledgeSnapshot.verificationRecommendation}
                  <div class="knowledge-verification">
                    <div class="verification-header">
                      <span
                        class={`verification-pill ${
                          knowledgeSnapshot.verificationRecommendation.recommend
                            ? 'verification-pill-recommend'
                            : 'verification-pill-pass'
                        }`}
                      >
                        {#if knowledgeSnapshot.verificationRecommendation.recommend}
                          Verification recommended
                        {:else}
                          Verification not required
                        {/if}
                      </span>
                      {#if knowledgeSnapshot.verificationRecommendation.recommend}
                        <button
                          class="verification-action"
                          on:click={requestVerificationMarket}
                          disabled={verificationMarketLoading}
                        >
                          {verificationMarketLoading ? 'Requesting…' : 'Request verification market'}
                        </button>
                      {/if}
                    </div>
                    <p class="verification-reason">
                      {knowledgeSnapshot.verificationRecommendation.reason}
                    </p>
                  </div>
                {/if}
              </div>
            {:else}
              <p class="knowledge-hint">
                No knowledge-graph claim found yet for this listing.
              </p>
            {/if}
          </div>

          <div class="trust-section" bind:this={trustSectionEl} id="trust" class:highlight={trustHighlight}>
            <div class="trust-header">
              <div class="trust-meta">
                <p class="eyebrow">Verifiable trust</p>
                {#if trustUpdatedAt}
                  <p class="trust-time">Updated {new Date(trustUpdatedAt).toLocaleTimeString()}</p>
                {/if}
              </div>
              {#if seller}
                <button
                  class="trust-refresh"
                  on:click={() => seller && loadReputation(seller.agent_id, true)}
                  disabled={trustLoading}
                >
                  {trustLoading ? 'Refreshing…' : 'Refresh trust'}
                </button>
              {/if}
            </div>
            {#if trustLoading}
              <div class="trust-loading">
                <div class="spinner small"></div>
                <p>Verifying trust graph…</p>
              </div>
            {:else if trustGraph}
              {#if trustUsingMock}
                <div class="trust-hint">Trust graph using preview data</div>
              {/if}
              {#if proofUpdateMessage}
                <div class="trust-hint success">{proofUpdateMessage}</div>
              {/if}
              <TrustGraph
                subject={trustGraph.subject}
                nodes={trustGraph.nodes}
                edges={trustGraph.edges}
                claims={trustGraph.claims}
                summary={trustGraph.summary}
                updatedAt={trustUpdatedAt || trustGraph.summary?.last_update || null}
                usingMock={trustUsingMock}
              />
              <ProofTrail items={proofTrail} />
            {:else if trustError}
              <div class="trust-error">Trust graph unavailable: {trustError}</div>
            {/if}
            <div class="proof-actions">
              <div class="proof-text">
                <p class="eyebrow">Proof request</p>
                <p class="proof-copy">
                  Ask the seller to attest this listing with a fresh claim or zk range proof.
                </p>
                <div class="proof-status">
                  <span class={`pill ${proofRequestStatus === 'requested' ? 'pill-success' : 'pill-neutral'}`}>
                    {#if proofRequestStatus === 'requested'}
                      Proof requested
                    {:else if proofRequestStatus === 'pending'}
                      Request pending
                    {:else}
                      No request yet
                    {/if}
                  </span>
                </div>
              </div>
              <button
                class="btn btn-secondary"
                on:click={requestProof}
                disabled={proofRequesting || trustLoading || !seller}
              >
                {#if proofRequesting}
                  Requesting…
                {:else}
                  Request proof
                {/if}
              </button>
            </div>
            {#if proofRequestMessage}
              <div class="proof-toast">{proofRequestMessage}</div>
            {/if}
            {#if listingRisk}
              <div class="risk-insight">
                <RiskInsightDrawer risk={listingRisk} proof={proofRequestStatus} open={false} />
              </div>
            {/if}
          </div>

          <!-- Purchase Actions -->
          <div class="purchase-section">
            <div class="quantity-selector">
              <label for="quantity-input">Quantity:</label>
              <div class="quantity-controls">
                <button
                  on:click={() => quantity = Math.max(1, quantity - 1)}
                  aria-label="Decrease quantity"
                >
                  −
                </button>
                <input
                  id="quantity-input"
                  type="number"
                  bind:value={quantity}
                  min="1"
                  max={listing.quantity_available || 99}
                  aria-label="Select quantity to purchase"
                />
                <button
                  on:click={() => quantity = Math.min(listing?.quantity_available || 99, quantity + 1)}
                  aria-label="Increase quantity"
                >
                  +
                </button>
              </div>
            </div>

            <div class="action-buttons">
              <button
                class="btn btn-secondary btn-large"
                on:click={() => addToCart()}
                disabled={!$isConnected}
                title={$isConnected ? '' : 'Connect to Holochain to add items'}
              >
                🛒 Add to Cart
              </button>
              <button
                class="btn btn-primary btn-large"
                on:click={buyNow}
                disabled={purchasing || !$isConnected}
                title={$isConnected ? '' : 'Connect to Holochain to purchase'}
              >
                {#if purchasing}
                  Processing...
                {:else}
                  ⚡ Buy Now
                {/if}
              </button>
            </div>
          </div>

          <!-- Description -->
          <div class="description-section">
            <h2>Description</h2>
            <p>{listing.description}</p>
          </div>

          <!-- Listing Meta -->
          <div class="meta-info">
            <p><strong>Listed:</strong> {formatDate(listing.created_at)}</p>
            {#if listing.views}
              <p><strong>Views:</strong> {listing.views}</p>
            {/if}
            <p><strong>Status:</strong> {listing.status}</p>
          </div>
        </div>
      </div>

      <!-- Reviews Section -->
      {#if reviews.length > 0}
        <div class="reviews-section">
          <h2>Reviews ({reviews.length})</h2>
          <div class="reviews-list">
            {#each reviews as review}
              <div class="review-card">
                <div class="review-header">
                  <div class="review-rating">
                    {'⭐'.repeat(review.rating)}
                  </div>
                  <span class="review-date">{formatDate(review.created_at)}</span>
                </div>
                <p class="review-comment">{review.comment}</p>
                <p class="review-author">— {review.reviewer_name}</p>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>

{#if showRiskModal}
  <div class="risk-modal" role="dialog" aria-modal="true" aria-label="Risk details">
    <div class="risk-modal-content">
      <button class="modal-close" on:click={() => (showRiskModal = false)}>✕</button>
      <h3>Why this risk?</h3>
      {#if listingRisk}
        <p class="modal-lede">
          We spotted signals that deserve a quick review. You can request proofs or ask the seller for more clarity.
        </p>
        <ul class="risk-flag-list">
          {#each listingRisk.flags as flag}
            <li>{flag}</li>
          {/each}
        </ul>
        <div class="modal-actions">
          <button class="btn btn-secondary" on:click={() => { showRiskModal = false; trustSectionEl?.scrollIntoView({ behavior: 'smooth' }); }}>
            Request proof
          </button>
          <button class="btn btn-primary" on:click={messageSeller}>Message seller</button>
        </div>
      {:else}
        <p>No risk signals.</p>
      {/if}
    </div>
  </div>
{/if}

<style>
  .listing-detail {
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

  .spinner.small {
    width: 28px;
    height: 28px;
    border-width: 3px;
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

  /* Breadcrumb */
  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 2rem;
    font-size: 0.875rem;
    color: #718096;
  }

  .breadcrumb button {
    background: none;
    border: none;
    color: #4299e1;
    cursor: pointer;
    padding: 0;
  }

  .breadcrumb button:hover {
    text-decoration: underline;
  }

  /* Main Grid */
  .listing-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    margin-bottom: 3rem;
  }

  /* Photos Section */
  .photos-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .main-photo {
    width: 100%;
    height: 500px;
    background: white;
    border-radius: 0.5rem;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    position: relative;
  }

  .preview-badge {
    position: absolute;
    top: 0.75rem;
    left: 0.75rem;
    background: rgba(255, 255, 255, 0.95);
    color: #dd6b20;
    border: 1px solid #fbd38d;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-weight: 700;
    font-size: 0.8rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  }

  .main-photo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .photo-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f7fafc;
    font-size: 5rem;
  }

  .photo-thumbnails {
    display: flex;
    gap: 0.75rem;
    overflow-x: auto;
  }

  .thumbnail {
    width: 100px;
    height: 100px;
    border: 2px solid transparent;
    border-radius: 0.375rem;
    overflow: hidden;
    cursor: pointer;
    flex-shrink: 0;
    background: white;
    padding: 0;
  }

  .thumbnail:hover {
    border-color: #4299e1;
  }

  .thumbnail.active {
    border-color: #4299e1;
  }

  .thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  /* Details Section */
  .details-section {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .title-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .title-row h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #2d3748;
    margin: 0;
    flex: 1;
  }

  .favorite-toggle {
    border: 1px solid #e2e8f0;
    background: white;
    border-radius: 999px;
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    font-size: 1.1rem;
    cursor: pointer;
    transition: transform 0.1s ease, box-shadow 0.1s ease, color 0.1s ease;
    color: #4a5568;
  }

  .favorite-toggle.active {
    color: #e53e3e;
  }

  .favorite-toggle:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
  }

  .price-section {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .price {
    font-size: 2.5rem;
    font-weight: 700;
    color: #38a169;
  }

  .availability {
    color: #718096;
    font-size: 0.875rem;
  }

  .category-tag {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #edf2f7;
    color: #4a5568;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
  }

  .risk-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  /* Seller Card */
  .seller-card {
    display: flex;
    gap: 1rem;
    padding: 1.5rem;
    background: #f7fafc;
    border-radius: 0.5rem;
    margin-bottom: 2rem;
  }

  .knowledge-section {
    margin-bottom: 1.5rem;
    padding: 1rem 1.25rem;
    border-radius: 0.5rem;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
  }

  .knowledge-header .eyebrow {
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366f1;
  }

  .knowledge-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
  }

  .knowledge-refresh {
    border: 1px solid #d1d5db;
    background: #ffffff;
    border-radius: 999px;
    padding: 0.25rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #374151;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .knowledge-refresh:hover:not(:disabled) {
    border-color: #6366f1;
    color: #4338ca;
    box-shadow: 0 4px 10px rgba(99, 102, 241, 0.16);
  }

  .knowledge-refresh:disabled {
    opacity: 0.65;
    cursor: default;
  }

  .knowledge-loading {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: #4b5563;
  }

  .knowledge-error {
    padding: 0.5rem 0.75rem;
    background: #fff5f5;
    color: #b91c1c;
    border: 1px solid #fecaca;
    border-radius: 0.5rem;
    font-size: 0.875rem;
  }

  .knowledge-card {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .knowledge-title {
    margin: 0;
    font-size: 0.95rem;
    color: #111827;
  }

  .knowledge-title span {
    font-weight: 600;
  }

  .knowledge-metric {
    margin: 0.15rem 0 0;
    font-size: 0.9rem;
    color: #4b5563;
  }

  .knowledge-epistemic {
    margin-top: 0.35rem;
    border-top: 1px dashed #e5e7eb;
    padding-top: 0.35rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .axis-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
  }

  .axis-label {
    color: #6b7280;
  }

  .axis-value {
    color: #111827;
    font-weight: 500;
  }

  .knowledge-verification {
    margin-top: 0.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .verification-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .verification-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .verification-pill-recommend {
    background: #fef3c7;
    color: #92400e;
  }

  .verification-pill-pass {
    background: #ecfdf3;
    color: #166534;
  }

  .verification-action {
    border: 1px solid #d1d5db;
    background: #ffffff;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #374151;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .verification-action:hover:not(:disabled) {
    border-color: #6366f1;
    color: #4338ca;
    box-shadow: 0 4px 10px rgba(99, 102, 241, 0.16);
  }

  .verification-action:disabled {
    opacity: 0.65;
    cursor: default;
  }

  .verification-reason {
    margin: 0;
    font-size: 0.85rem;
    color: #4b5563;
  }

  .knowledge-hint {
    margin: 0;
    font-size: 0.9rem;
    color: #6b7280;
  }

  .trust-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 2rem;
    scroll-margin-top: 90px;
  }

  .trust-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
  }

  .trust-meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .trust-meta .eyebrow {
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6366f1;
  }

  .trust-time {
    margin: 0;
    color: #4a5568;
    font-size: 0.9rem;
  }

  .trust-refresh {
    border: 1px solid #cbd5e0;
    background: #ffffff;
    border-radius: 0.5rem;
    padding: 0.5rem 0.9rem;
    font-weight: 700;
    color: #1e293b;
    cursor: pointer;
    transition: all 0.2s;
  }

  .trust-refresh:hover:not(:disabled) {
    border-color: #6366f1;
    color: #4338ca;
    box-shadow: 0 6px 14px rgba(67, 56, 202, 0.12);
  }

  .trust-refresh:disabled {
    opacity: 0.65;
    cursor: not-allowed;
  }

  .trust-loading {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: #4a5568;
  }

  .trust-hint {
    padding: 0.5rem 0.75rem;
    background: #f0f9ff;
    color: #0369a1;
    border: 1px solid #bae6fd;
    border-radius: 0.5rem;
    font-weight: 600;
  }

  .trust-hint.success {
    background: #ecfdf3;
    color: #166534;
    border-color: #bbf7d0;
  }

  .trust-error {
    padding: 0.5rem 0.75rem;
    background: #fff5f5;
    color: #c53030;
    border: 1px solid #fed7d7;
    border-radius: 0.5rem;
    font-weight: 600;
  }

  .trust-section.highlight {
    outline: 2px solid #6366f1;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.18);
    border-radius: 0.75rem;
  }

  .risk-modal {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.35);
    display: grid;
    place-items: center;
    z-index: 9999;
    padding: 1rem;
  }

  .risk-modal-content {
    background: #ffffff;
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    max-width: 520px;
    width: 100%;
    position: relative;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2);
  }

  .risk-modal h3 {
    margin-top: 0;
    margin-bottom: 0.35rem;
    color: #111827;
  }

  .modal-lede {
    margin: 0 0 0.75rem;
    color: #475569;
  }

  .risk-flag-list {
    margin: 0 0 1rem;
    padding-left: 1.2rem;
    color: #1f2937;
  }

  .modal-actions {
    display: flex;
    gap: 0.75rem;
    justify-content: flex-end;
  }

  .proof-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    background: #f8fafc;
    flex-wrap: wrap;
  }

  .proof-text {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .proof-copy {
    margin: 0;
    color: #475569;
  }

  .proof-status {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .pill {
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid transparent;
  }

  .pill-neutral {
    background: #edf2f7;
    color: #1a202c;
    border-color: #e2e8f0;
  }

  .pill-success {
    background: #dcfce7;
    color: #166534;
    border-color: #bbf7d0;
  }

  .proof-toast {
    margin-top: 0.5rem;
    padding: 0.6rem 0.75rem;
    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
    border-radius: 0.5rem;
    font-weight: 600;
  }

  .seller-avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    overflow: hidden;
    flex-shrink: 0;
  }

  .seller-avatar img {
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
    font-size: 1.5rem;
    font-weight: 700;
  }

  .seller-info {
    flex: 1;
  }

  .seller-name {
    background: none;
    border: none;
    font-size: 1.125rem;
    font-weight: 600;
    color: #4299e1;
    cursor: pointer;
    padding: 0;
    margin-bottom: 0.25rem;
  }

  .seller-name:hover {
    text-decoration: underline;
  }

  .seller-stats {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.25rem;
  }

  .trust-score {
    color: #38a169;
    font-weight: 600;
    font-size: 0.875rem;
  }

  .rating {
    color: #718096;
    font-size: 0.875rem;
  }

  .member-since {
    color: #a0aec0;
    font-size: 0.75rem;
  }

  /* Purchase Section */
  .purchase-section {
    padding: 1.5rem 0;
    border-top: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 2rem;
  }

  .quantity-selector {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .quantity-selector label {
    font-weight: 600;
    color: #2d3748;
  }

  .quantity-controls {
    display: flex;
    align-items: center;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    overflow: hidden;
  }

  .quantity-controls button {
    padding: 0.5rem 1rem;
    background: #f7fafc;
    border: none;
    cursor: pointer;
    font-size: 1.25rem;
    font-weight: 700;
  }

  .quantity-controls button:hover {
    background: #e2e8f0;
  }

  .quantity-controls input {
    width: 60px;
    padding: 0.5rem;
    border: none;
    text-align: center;
    font-size: 1rem;
  }

  .action-buttons {
    display: flex;
    gap: 1rem;
  }

  .btn {
    flex: 1;
    padding: 1rem 1.5rem;
    border: none;
    border-radius: 0.375rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }

  .btn-primary {
    background: #4299e1;
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: #3182ce;
  }

  .btn-secondary {
    background: #e2e8f0;
    color: #2d3748;
  }

  .btn-secondary:hover {
    background: #cbd5e0;
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Description */
  .description-section {
    margin-bottom: 2rem;
  }

  .description-section h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.75rem;
  }

  .description-section p {
    color: #4a5568;
    line-height: 1.7;
    white-space: pre-wrap;
  }

  /* Meta Info */
  .meta-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .meta-info p {
    font-size: 0.875rem;
    color: #718096;
  }

  .meta-info strong {
    color: #4a5568;
  }

  /* Reviews */
  .reviews-section {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .reviews-section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1.5rem;
  }

  .reviews-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .review-card {
    padding: 1.5rem;
    background: #f7fafc;
    border-radius: 0.5rem;
  }

  .review-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .review-rating {
    color: #ecc94b;
    font-size: 1.125rem;
  }

  .review-date {
    font-size: 0.875rem;
    color: #a0aec0;
  }

  .review-comment {
    color: #4a5568;
    line-height: 1.6;
    margin-bottom: 0.5rem;
  }

  .review-author {
    font-size: 0.875rem;
    color: #718096;
    font-style: italic;
  }

  /* Responsive */
  @media (max-width: 968px) {
    .listing-grid {
      grid-template-columns: 1fr;
      gap: 2rem;
    }

    .action-buttons {
      flex-direction: column;
    }

    .details-section h1 {
      font-size: 1.5rem;
    }

    .price {
      font-size: 2rem;
    }
  }
</style>
