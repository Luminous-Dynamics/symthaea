<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { goto } from '$app/navigation';
  import { cartItems, itemCount, subtotal, tax, shipping, total } from '$lib/stores/cart';
  import { notifications } from '$lib/stores';
  import { getProofStatus } from '$lib/stores';
  import { analyzeRisk } from '$lib/risk';
  import type { RiskSignal } from '$types';
  import RiskInsightDrawer from '$lib/components/RiskInsightDrawer.svelte';

  /**
   * Remove item from cart
   */
  function handleRemoveItem(listingHash: string, title: string) {
    cartItems.removeItem(listingHash);
    notifications.success('Item Removed', `${title} removed from cart`);
  }

  /**
   * Update item quantity
   */
  function handleUpdateQuantity(listingHash: string, newQuantity: number) {
    if (newQuantity < 1) {
      return; // Don't allow quantity below 1
    }

    cartItems.updateQuantity(listingHash, newQuantity);
  }

  /**
   * Increment quantity
   */
  function incrementQuantity(listingHash: string, currentQuantity: number) {
    handleUpdateQuantity(listingHash, currentQuantity + 1);
  }

  /**
   * Decrement quantity
   */
  function decrementQuantity(listingHash: string, currentQuantity: number) {
    if (currentQuantity > 1) {
      handleUpdateQuantity(listingHash, currentQuantity - 1);
    }
  }

  /**
   * Continue shopping
   */
  function continueShopping() {
    goto('/browse');
  }

  /**
   * Proceed to checkout with guardrails
   */
  function proceedToCheckout() {
    const hasHighRisk = $cartItems.some((item) => (riskLookup[item.listing_hash]?.score || 0) > 0.5);
    const hasProofPending = $cartItems.some(
      (item) => proofLookup[item.listing_hash] === 'requested' || proofLookup[item.listing_hash] === 'pending'
    );

    if (safeMode && (hasHighRisk || hasProofPending)) {
      notifications.warning(
        'Safe mode blocking checkout',
        'High-risk or proof-pending items present. Turn off safe mode to continue.'
      );
      return;
    }

    if (hasHighRisk || hasProofPending) {
      notifications.warning(
        'Review recommended',
        'Some items are high-risk or missing proof. Confirm to continue.'
      );
    }

    goto('/checkout');
  }

  /**
   * Format currency
   */
  function formatCurrency(amount: number): string {
    return `$${amount.toFixed(2)}`;
  }

  /**
   * Get IPFS image URL
   */
  function getImageUrl(cid: string): string {
    return `https://ipfs.io/ipfs/${cid}`;
  }

  let riskLookup: Record<string, RiskSignal> = {};
  let proofLookup: Record<string, string> = {};
  let safeMode = true;
  $: flaggedItems =
    $cartItems?.filter(
      (item) =>
        (riskLookup[item.listing_hash]?.score || 0) > 0.5 ||
        proofLookup[item.listing_hash] === 'requested' ||
        proofLookup[item.listing_hash] === 'pending'
    ) || [];

  $: hydrateProofLookup();
  $: hydrateRiskLookup();

  function hydrateProofLookup() {
    const next: Record<string, string> = {};
    $cartItems?.forEach((item) => {
      const status = getProofStatus(item.seller_agent_id, item.listing_hash);
      if (status) next[item.listing_hash] = status;
    });
    proofLookup = next;
  }

  async function hydrateRiskLookup() {
    const listingsLike = $cartItems?.map((item) => ({
      id: item.listing_hash,
      listing_hash: item.listing_hash,
      title: item.title,
      description: item.title,
      price: item.price,
      category: 'Other',
      photos_ipfs_cids: item.photo_cid ? [item.photo_cid] : [],
      seller_agent_id: item.seller_agent_id,
      created_at: Date.now(),
      status: 'active',
    })) as any[];
    if (!listingsLike || listingsLike.length === 0) {
      riskLookup = {};
      return;
    }
    riskLookup = await analyzeRisk(listingsLike);
  }
</script>

<main class="cart-page">
  <div class="container">
    <!-- Header -->
    <header class="page-header">
      <h1>Shopping Cart</h1>
      <p class="item-count">
        {#if $itemCount === 0}
          Your cart is empty
        {:else if $itemCount === 1}
          1 item
        {:else}
          {$itemCount} items
        {/if}
      </p>
    </header>

    {#if $cartItems.length === 0}
      <!-- Empty Cart State -->
      <div class="empty-cart">
        <svg
          class="empty-cart-icon"
          xmlns="http://www.w3.org/2000/svg"
          width="120"
          height="120"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="9" cy="21" r="1" />
          <circle cx="20" cy="21" r="1" />
          <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6" />
        </svg>

        <h2>Your cart is empty</h2>
        <p>Add some items to get started!</p>

        <button class="btn btn-primary" on:click={continueShopping}>Browse Marketplace</button>
      </div>
    {:else}
      {#if flaggedItems.length}
        <div class="safe-banner">
          <div>
            <strong>Safe mode {safeMode ? 'on' : 'off'}.</strong>
            <span>
              {flaggedItems.length} item{flaggedItems.length === 1 ? '' : 's'} marked high-risk or proof-pending.
            </span>
          </div>
          <div class="safe-actions">
            <label class="toggle">
              <input type="checkbox" bind:checked={safeMode} />
              <span>Safe mode</span>
            </label>
            {#if safeMode}
              <small>Turn off to proceed anyway.</small>
            {/if}
          </div>
        </div>
      {/if}
      <!-- Cart Content -->
      <div class="cart-content">
        <!-- Cart Items -->
        <div class="cart-items">
          {#each $cartItems as item (item.listing_hash)}
            <div class="cart-item">
              <!-- Item Image -->
              <div class="item-image">
                {#if item.photo_cid}
                  <img src={getImageUrl(item.photo_cid)} alt={item.title} />
                {:else}
                  <div class="no-image">No Image</div>
                {/if}
              </div>

              <div class="item-flags">
                {#if proofLookup[item.listing_hash]}
                  <span
                    class={`pill ${proofLookup[item.listing_hash]}`}
                    title={`Proof ${proofLookup[item.listing_hash]}`}
                  >
                    Proof {proofLookup[item.listing_hash]}
                  </span>
                {/if}
                {#if riskLookup[item.listing_hash]?.score > 0.5}
                  <span class="pill risk" title={riskLookup[item.listing_hash]?.flags?.[0]}>
                    Review suggested
                  </span>
                {/if}
              </div>

              <!-- Item Details -->
              <div class="item-details">
                <h3 class="item-title">{item.title}</h3>
                <p class="item-seller">Sold by {item.seller_name}</p>
                <p class="item-price">{formatCurrency(item.price)} each</p>
                {#if riskLookup[item.listing_hash]}
                  <RiskInsightDrawer
                    risk={riskLookup[item.listing_hash]}
                    proof={proofLookup[item.listing_hash] || null}
                    open={false}
                  />
                {/if}
              </div>

              <!-- Item Quantity -->
              <div class="item-quantity">
                <label for="quantity-{item.listing_hash}">Quantity</label>
                <div class="quantity-controls">
                  <button
                    class="quantity-btn"
                    on:click={() => decrementQuantity(item.listing_hash, item.quantity)}
                    disabled={item.quantity <= 1}
                    aria-label="Decrease quantity"
                  >
                    −
                  </button>
                  <input
                    id="quantity-{item.listing_hash}"
                    type="number"
                    class="quantity-input"
                    value={item.quantity}
                    min="1"
                    aria-label="Quantity for {item.title}"
                    on:change={(e) =>
                      handleUpdateQuantity(item.listing_hash, parseInt(e.currentTarget.value) || 1)}
                  />
                  <button
                    class="quantity-btn"
                    on:click={() => incrementQuantity(item.listing_hash, item.quantity)}
                    aria-label="Increase quantity"
                  >
                    +
                  </button>
                </div>
              </div>

              <!-- Item Total -->
              <div class="item-total">
                <span class="item-total-label">Total</span>
                <p class="total-price">{formatCurrency(item.price * item.quantity)}</p>
              </div>

              <!-- Remove Button -->
              <button
                class="remove-btn"
                on:click={() => handleRemoveItem(item.listing_hash, item.title)}
                title="Remove item"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <polyline points="3 6 5 6 21 6" />
                  <path
                    d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"
                  />
                  <line x1="10" y1="11" x2="10" y2="17" />
                  <line x1="14" y1="11" x2="14" y2="17" />
                </svg>
              </button>
            </div>
          {/each}
        </div>

        <!-- Order Summary -->
        <div class="order-summary">
          <h2>Order Summary</h2>

          <div class="summary-row">
            <span>Subtotal ({$itemCount} {$itemCount === 1 ? 'item' : 'items'})</span>
            <span>{formatCurrency($subtotal)}</span>
          </div>

          <div class="summary-row">
            <span>Shipping</span>
            <span>{formatCurrency($shipping)}</span>
          </div>

          <div class="summary-row">
            <span>Tax (8%)</span>
            <span>{formatCurrency($tax)}</span>
          </div>

          <div class="summary-divider"></div>

          <div class="summary-row summary-total">
            <span>Total</span>
            <span>{formatCurrency($total)}</span>
          </div>

          {#if flaggedItems.length}
            <div class="safe-summary">
              <div>
                <p class="safe-label">Safe mode is {safeMode ? 'blocking' : 'off'}.</p>
                <p class="safe-copy">
                  {flaggedItems.length} item{flaggedItems.length === 1 ? '' : 's'} marked high-risk or proof-pending.
                </p>
                {#if !safeMode}
                  <p class="safe-warning">Proceeding without safe mode. Double-check trust before checkout.</p>
                  <ul class="flagged-list">
                    {#each flaggedItems as flagged}
                      <li>
                        <span class="flagged-title">{flagged.title}</span>
                        <span class="flagged-pill" title={riskLookup[flagged.listing_hash]?.flags?.[0] || 'Trust check required'}>
                          {(riskLookup[flagged.listing_hash]?.flags?.[0] || 'Trust check required')}
                        </span>
                        {#if proofLookup[flagged.listing_hash]}
                          <span class={`flagged-pill proof ${proofLookup[flagged.listing_hash]}`}>
                            Proof {proofLookup[flagged.listing_hash]}
                          </span>
                        {/if}
                      </li>
                    {/each}
                  </ul>
                {/if}
              </div>
              {#if safeMode}
                <button class="btn btn-secondary" on:click={() => (safeMode = false)}>
                  Proceed anyway
                </button>
              {/if}
            </div>
          {/if}
          {#if safeMode && flaggedItems.length === 0}
            <p class="safe-hint">Safe mode on: no trust issues detected in this cart.</p>
          {:else if !safeMode}
            <div class="trust-summary">
              <p class="safe-label">Trust summary (safe mode off)</p>
              {#if flaggedItems.length}
                <ul class="flagged-list">
                  {#each flaggedItems as flagged}
                    <li>
                      <span class="flagged-title">{flagged.title}</span>
                      <span class="flagged-pill" title={riskLookup[flagged.listing_hash]?.flags?.[0] || 'Trust check required'}>
                        {(riskLookup[flagged.listing_hash]?.flags?.[0] || 'Trust check required')}
                      </span>
                      {#if proofLookup[flagged.listing_hash]}
                        <span class={`flagged-pill proof ${proofLookup[flagged.listing_hash]}`}>
                          Proof {proofLookup[flagged.listing_hash]}
                        </span>
                      {/if}
                    </li>
                  {/each}
                </ul>
              {:else}
                <p class="safe-copy all-clear">No flagged items, but safe mode disabled.</p>
              {/if}
            </div>
          {/if}

          <button
            class="btn btn-primary btn-checkout"
            on:click={proceedToCheckout}
            disabled={safeMode && flaggedItems.length > 0}
            title={safeMode && flaggedItems.length > 0
              ? 'Turn off safe mode to proceed with high-risk items'
              : 'Proceed to checkout'}
          >
            Proceed to Checkout
          </button>

          <button class="btn btn-secondary" on:click={continueShopping}>Continue Shopping</button>
        </div>
      </div>
    {/if}
  </div>
</main>

<style>
  .cart-page {
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1rem;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
  }

  .page-header {
    text-align: center;
    color: white;
    margin-bottom: 2rem;
  }

  .safe-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fffaf0;
    border: 1px solid #f6ad55;
    color: #9a3412;
    padding: 0.9rem 1rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    gap: 0.75rem;
  }

  .safe-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .safe-actions small {
    color: #b7791f;
    font-weight: 600;
  }

  .safe-summary {
    margin: 1rem 0;
    padding: 0.9rem 1rem;
    background: #fffaf0;
    border: 1px solid #f6ad55;
    color: #9a3412;
    border-radius: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
  }

  .safe-label {
    margin: 0;
    font-weight: 700;
  }

  .safe-copy {
    margin: 0;
    color: #b7791f;
  }

  .safe-copy.all-clear {
    color: #38a169;
    font-weight: 700;
  }

  .safe-warning {
    margin: 0.25rem 0 0;
    color: #c05621;
    font-weight: 700;
  }

  .flagged-list {
    margin: 0.5rem 0 0;
    padding-left: 1.2rem;
    color: #9a3412;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .flagged-list li {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }

  .flagged-title {
    font-weight: 700;
  }

  .flagged-pill {
    background: #fff7ed;
    color: #9a3412;
    border: 1px solid #fdba74;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.8rem;
    white-space: nowrap;
  }

  .flagged-pill.proof.fulfilled {
    background: #dcfce7;
    border-color: #86efac;
    color: #166534;
  }

  .flagged-pill.proof.pending,
  .flagged-pill.proof.requested {
    background: #fef3c7;
    border-color: #fcd34d;
    color: #92400e;
  }

  .flagged-pill.proof.denied {
    background: #fee2e2;
    border-color: #fca5a5;
    color: #991b1b;
  }

  .safe-hint {
    margin: 0.5rem 0 0;
    color: #b7791f;
    font-size: 0.9rem;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-weight: 700;
  }

  .page-header h1 {
    font-size: 2.5rem;
    margin: 0 0 0.5rem 0;
  }

  .item-count {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0;
  }

  /* Empty Cart */
  .empty-cart {
    background: white;
    border-radius: 12px;
    padding: 4rem 2rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
  }

  .empty-cart-icon {
    color: #cbd5e0;
    margin-bottom: 2rem;
  }

  .empty-cart h2 {
    font-size: 1.8rem;
    color: #2d3748;
    margin: 0 0 1rem 0;
  }

  .empty-cart p {
    color: #718096;
    font-size: 1.1rem;
    margin: 0 0 2rem 0;
  }

  /* Cart Content */
  .cart-content {
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 2rem;
  }

  /* Cart Items */
  .cart-items {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .cart-item {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    display: grid;
    grid-template-columns: 120px 1fr auto auto auto;
    gap: 1.5rem;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
  }

  .cart-item:hover {
    transform: translateY(-2px);
  }

  .item-image {
    width: 120px;
    height: 120px;
    border-radius: 8px;
    overflow: hidden;
    background: #f7fafc;
  }

  .item-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .no-image {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #a0aec0;
    font-size: 0.875rem;
  }

  .item-details {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .item-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2d3748;
    margin: 0;
  }

  .item-seller {
    font-size: 0.9rem;
    color: #718096;
    margin: 0;
  }

  .item-price {
    font-size: 1rem;
    font-weight: 600;
    color: #667eea;
    margin: 0;
  }

  .risk-insight {
    margin-top: 0.5rem;
    max-width: 360px;
  }

  .item-quantity {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .item-quantity label {
    font-size: 0.875rem;
    color: #718096;
    font-weight: 600;
  }

  .quantity-controls {
    display: flex;
    align-items: center;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
  }

  .quantity-btn {
    width: 36px;
    height: 36px;
    border: none;
    background: #f7fafc;
    color: #2d3748;
    font-size: 1.2rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }

  .quantity-btn:hover:not(:disabled) {
    background: #e2e8f0;
  }

  .quantity-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .quantity-input {
    width: 50px;
    height: 36px;
    border: none;
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    color: #2d3748;
    appearance: textfield;
    -moz-appearance: textfield;
  }

  .quantity-input::-webkit-outer-spin-button,
  .quantity-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }

  .item-total {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 100px;
  }

  .item-total-label {
    font-size: 0.875rem;
    color: #718096;
    font-weight: 600;
  }

  .total-price {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2d3748;
    margin: 0;
  }

  .remove-btn {
    width: 40px;
    height: 40px;
    border: none;
    background: #fff5f5;
    color: #e53e3e;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .remove-btn:hover {
    background: #e53e3e;
    color: white;
  }

  .item-flags {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .pill {
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.8rem;
    border: 1px solid transparent;
    text-transform: capitalize;
  }

  .pill.fulfilled {
    background: #dcfce7;
    color: #166534;
    border-color: #bbf7d0;
  }

  .pill.pending,
  .pill.requested {
    background: #eef2ff;
    color: #4338ca;
    border-color: #c7d2fe;
  }

  .pill.risk {
    background: #fff7ed;
    color: #c2410c;
    border-color: #fed7aa;
  }

  /* Order Summary */
  .order-summary {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    height: fit-content;
    position: sticky;
    top: 2rem;
  }

  .order-summary h2 {
    font-size: 1.5rem;
    color: #2d3748;
    margin: 0 0 1.5rem 0;
  }

  .summary-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 1rem;
    font-size: 1rem;
    color: #4a5568;
  }

  .summary-divider {
    height: 2px;
    background: #e2e8f0;
    margin: 1.5rem 0;
  }

  .summary-total {
    font-size: 1.3rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 2rem;
  }

  /* Buttons */
  .btn {
    width: 100%;
    padding: 1rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
    margin-bottom: 1rem;
  }

  .btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }

  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
  }

  .btn-checkout {
    margin-bottom: 1rem;
  }

  .btn-secondary {
    background: white;
    color: #2d3748;
    border: 2px solid #e2e8f0;
  }

  .btn-secondary:hover {
    background: #f7fafc;
  }

  /* Responsive */
  @media (max-width: 1024px) {
    .cart-content {
      grid-template-columns: 1fr;
    }

    .order-summary {
      position: static;
    }

    .cart-item {
      grid-template-columns: 100px 1fr;
      grid-template-rows: auto auto auto;
      gap: 1rem;
    }

    .item-image {
      width: 100px;
      height: 100px;
      grid-row: 1 / 3;
    }

    .item-details {
      grid-column: 2 / 3;
    }

    .item-quantity,
    .item-total {
      grid-column: 2 / 3;
    }

    .remove-btn {
      grid-column: 1 / 3;
      justify-self: center;
      width: 100%;
      max-width: 200px;
    }
  }

  @media (max-width: 640px) {
    .page-header h1 {
      font-size: 2rem;
    }

    .cart-item {
      padding: 1rem;
    }

    .order-summary {
      padding: 1.5rem;
    }
  }
</style>
