<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * Checkout Flow Page
   *
   * Complete checkout process with:
   * - Order summary from cart
   * - Shipping address form
   * - Payment method selection
   * - Order review and confirmation
   * - Holochain transaction creation
   * - Post-checkout redirect
   *
   * This demonstrates:
   * - Multi-step checkout process
   * - Form validation
   * - Payment integration placeholders
   * - Transaction creation for each cart item
   * - Cart clearing after success
   */

  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { get } from 'svelte/store';
  import { getIpfsUrl } from '$lib/ipfs/ipfsClient';
  import { cartItems as cart, subtotal, tax, shipping, total, isConnected } from '$lib/stores';
  import { notifications, getProofStatus } from '$lib/stores';
  import { analyzeRisk } from '$lib/risk';
  import type { RiskSignal } from '$types';
  import { recordGuardrailOverride, attachTransactions } from '$lib/stores/guardrails';
  import { initHolochainClient } from '$lib/holochain';
  import { createTransaction } from '$lib/holochain/transactions';
  import {
    verifyTransactionIdentity,
    getRequiredAssuranceLevel,
    isIdentityAvailable,
    checkIdentityConnection,
    type TransactionIdentityVerification,
    type AssuranceLevel,
    ASSURANCE_LEVEL_DESCRIPTION,
  } from '$lib/holochain/identity';
  import RiskInsightDrawer from '$lib/components/RiskInsightDrawer.svelte';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import type { PaymentMethod, CreateTransactionInput } from '$types';

  // Cart state (from store)
  let loading = true;
  let error = '';

  // Identity verification state
  let identityVerification: TransactionIdentityVerification | null = null;
  let identityWarnings: string[] = [];
  let identityCheckComplete = false;
  let requiredAssuranceLevel: AssuranceLevel = 'E0';

  // Checkout step (1: Shipping, 2: Payment, 3: Review)
  let currentStep = 1;
  let riskLookup: Record<string, RiskSignal> = {};
  let proofLookup: Record<string, string> = {};
  let riskSnapshotHash: string | null = null;
  let guardrailWarning = '';
  let guardrailHardBlock = false;
  let guardrailAcknowledged = false;
  let guardrailOverrideNote = '';
  let guardrailLogId: string | null = null;
  const MIN_OVERRIDE_NOTE = 12;
  $: canOverride = guardrailOverrideNote.trim().length >= MIN_OVERRIDE_NOTE;
  $: if ($cart) {
    // Reset guardrail state when cart contents change
    guardrailAcknowledged = false;
    guardrailOverrideNote = '';
    guardrailWarning = '';
    guardrailHardBlock = false;
    guardrailLogId = null;
  }

  // Shipping form
  let shippingAddress = {
    name: '',
    address_line_1: '',
    address_line_2: '',
    city: '',
    state: '',
    postal_code: '',
    country: 'USA',
    phone: '',
  };

  // Payment form
  let paymentMethod: PaymentMethod = 'crypto';
  let paymentDetails = {
    // Crypto
    cryptoCurrency: 'ETH',
    walletAddress: '',
    // Fiat
    cardNumber: '',
    cardName: '',
    cardExpiry: '',
    cardCVV: '',
  };

  // Checkout state
  let processingCheckout = false;
  let checkoutError = '';
  let checkoutSuccess = false;
  let transactionIds: string[] = [];

  /**
   * Load cart from store
   */
  onMount(() => {
    loading = true;

    try {
      // Get cart items from store
      const items = get(cart);

      if (items.length === 0) {
        notifications.warning('Cart empty', 'Add items before checking out');
      } else {
        notifications.info('Checkout', `${items.length} items in cart`);
      }
      hydrateGuards(items);
    } catch (e: any) {
      error = e.message || 'Failed to load cart';
      notifications.error('Failed to load cart', error);
    } finally {
      loading = false;
    }
  });

  /**
   * Validate shipping address
   */
  function validateShipping(): boolean {
    if (!shippingAddress.name.trim()) {
      error = 'Full name is required';
      notifications.error('Validation error', 'Full name is required');
      return false;
    }
    if (!shippingAddress.address_line_1.trim()) {
      error = 'Address is required';
      notifications.error('Validation error', 'Address is required');
      return false;
    }
    if (!shippingAddress.city.trim()) {
      error = 'City is required';
      notifications.error('Validation error', 'City is required');
      return false;
    }
    if (!shippingAddress.state.trim()) {
      error = 'State is required';
      notifications.error('Validation error', 'State is required');
      return false;
    }
    if (!shippingAddress.postal_code.trim()) {
      error = 'ZIP code is required';
      notifications.error('Validation error', 'ZIP code is required');
      return false;
    }
    if (!shippingAddress.phone.trim()) {
      error = 'Phone number is required';
      notifications.error('Validation error', 'Phone number is required');
      return false;
    }
    return true;
  }

  /**
   * Validate payment details
   */
  function validatePayment(): boolean {
    if (paymentMethod === 'crypto') {
      if (!paymentDetails.walletAddress.trim()) {
        error = 'Wallet address is required';
        notifications.error('Validation error', 'Wallet address is required');
        return false;
      }
    } else if (paymentMethod === 'credit_card') {
      if (!paymentDetails.cardNumber.trim()) {
        error = 'Card number is required';
        notifications.error('Validation error', 'Card number is required');
        return false;
      }
      if (!paymentDetails.cardName.trim()) {
        error = 'Cardholder name is required';
        notifications.error('Validation error', 'Cardholder name is required');
        return false;
      }
      if (!paymentDetails.cardExpiry.trim()) {
        error = 'Expiry date is required';
        notifications.error('Validation error', 'Expiry date is required');
        return false;
      }
      if (!paymentDetails.cardCVV.trim()) {
        error = 'CVV is required';
        notifications.error('Validation error', 'CVV is required');
        return false;
      }
    }
    return true;
  }

  /**
   * Move to next step
   */
  async function nextStep() {
    error = '';
    guardrailWarning = '';

    if (currentStep === 1) {
      if (!validateShipping()) return;
      currentStep = 2;
    } else if (currentStep === 2) {
      if (!validatePayment()) return;
      currentStep = 3;

      // Trigger identity verification when moving to review step
      const items = get(cart);
      if (items.length > 0) {
        await verifyIdentityForCheckout(items);
      }
    }
  }

  /**
   * Move to previous step
   */
  function previousStep() {
    error = '';
    guardrailWarning = '';
    if (currentStep > 1) {
      currentStep--;
    }
  }

  /**
   * Complete checkout
   */
  async function completeCheckout() {
    processingCheckout = true;
    checkoutError = '';
    error = '';
    guardrailWarning = '';
    guardrailHardBlock = false;

    const items = get(cart);

    if (items.length === 0) {
      error = 'Your cart is empty';
      notifications.warning('Cart empty', 'Add items before checking out');
      processingCheckout = false;
      return;
    }

    await hydrateGuards(items);

    const hasHighRisk = items.some((item) => (riskLookup[item.listing_hash]?.score || 0) > 0.5);
    const hasProofPending = items.some(
      (item) =>
        proofLookup[item.listing_hash] === 'requested' || proofLookup[item.listing_hash] === 'pending'
    );

    if (hasHighRisk || hasProofPending) {
      if (!guardrailAcknowledged) {
        guardrailWarning = 'Review recommended: high-risk or proof-pending items.';
        guardrailHardBlock = true;
        guardrailOverrideNote = '';
        notifications.warning('Review recommended', guardrailWarning);
        processingCheckout = false;
        return;
      }
      guardrailWarning = 'Override noted. Proceeding despite trust flags on this order.';
      notifications.info('Override in effect', guardrailWarning);
    }

    try {
      // Initialize Holochain client
      const client = await initHolochainClient();

      notifications.info('Processing', `Creating ${items.length} transactions...`);

      // Create transactions for each cart item
      const transactions = await Promise.all(
        items.map(async (item) => {
          const transactionInput: CreateTransactionInput = {
            listing_hash: item.listing_hash,
            quantity: item.quantity,
            shipping_address: shippingAddress,
            payment_method: paymentMethod,
            trust_override_note: guardrailAcknowledged ? guardrailOverrideNote.trim() : undefined,
            trust_override_flags: guardrailAcknowledged
              ? Object.fromEntries(
                  Object.entries(riskLookup).map(([key, risk]) => [key, risk?.flags?.[0]])
                )
              : undefined,
            risk_snapshot_hash: riskSnapshotHash || undefined,
          };

          const transaction = await createTransaction(client, transactionInput);
          console.log('Transaction created:', transaction);
          return transaction;
        })
      );

      // Store transaction IDs
      transactionIds = transactions.map((t) => t.id);
      if (guardrailLogId && transactionIds.length) {
        attachTransactions(guardrailLogId, transactionIds);
      }

      // Clear cart after successful checkout
      cart.clear();

      // Success
      checkoutSuccess = true;
      notifications.success(
        'Order placed!',
        `${transactions.length} transactions created successfully`
      );

      // Redirect to transaction tracking after 3 seconds
      setTimeout(() => {
        goto('/transactions?trust=overrides');
      }, 3000);
    } catch (e: any) {
      checkoutError = e.message || 'Failed to complete checkout';
      notifications.error('Checkout failed', checkoutError);
      console.error('Error completing checkout:', e);
    } finally {
      processingCheckout = false;
    }
  }

  async function hydrateGuards(items: any[]) {
    const listingsLike = items.map((item) => ({
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
    riskLookup = await analyzeRisk(listingsLike);
    const snapshot = JSON.stringify(
      listingsLike.map((l) => ({
        listing_hash: l.listing_hash,
        risk: riskLookup[l.listing_hash || l.id],
      }))
    );
    riskSnapshotHash = btoa(unescape(encodeURIComponent(snapshot))).slice(0, 64);
    const proof: Record<string, string> = {};
    items.forEach((item) => {
      const status = getProofStatus(item.seller_agent_id, item.listing_hash);
      if (status) proof[item.listing_hash] = status;
    });
    proofLookup = proof;

    // Calculate required assurance level based on total value
    const totalValue = items.reduce((sum, item) => sum + item.price * item.quantity, 0);
    requiredAssuranceLevel = getRequiredAssuranceLevel(totalValue, 'USD');
  }

  /**
   * Verify identity for the transaction
   * Called when moving to review step
   */
  async function verifyIdentityForCheckout(items: any[]) {
    identityCheckComplete = false;
    identityWarnings = [];

    try {
      const client = await initHolochainClient();

      // Check if identity service is available
      await checkIdentityConnection();

      if (!isIdentityAvailable()) {
        identityWarnings.push('Identity verification running in degraded mode');
        identityCheckComplete = true;
        return;
      }

      // Get buyer DID (current user)
      const appInfo = await client.appInfo();
      const agentKey = appInfo?.agent_pub_key;
      const buyerDid = agentKey ? `did:mycelix:${Buffer.from(agentKey).toString('base64')}` : '';

      // Verify identity for each seller
      const sellerDids = [...new Set(items.map((item) => `did:mycelix:${item.seller_agent_id}`))];
      const totalValue = items.reduce((sum, item) => sum + item.price * item.quantity, 0);

      // Verify all sellers
      for (const sellerDid of sellerDids) {
        const verification = await verifyTransactionIdentity(
          client,
          buyerDid,
          sellerDid,
          totalValue,
          'USD'
        );

        if (!identityVerification) {
          identityVerification = verification;
        }

        // Collect warnings
        identityWarnings = [...identityWarnings, ...verification.warnings];
      }

      // Notify if there are identity warnings
      if (identityWarnings.length > 0) {
        notifications.warning(
          'Identity Notice',
          `${identityWarnings.length} identity-related warning(s) for this transaction`
        );
      }
    } catch (e: any) {
      console.warn('Identity verification failed:', e);
      identityWarnings.push(`Identity check failed: ${e.message || 'Unknown error'}`);
    } finally {
      identityCheckComplete = true;
    }
  }

  function overrideGuardrail() {
    if (!canOverride) {
      notifications.warning('Add a short note', 'Acknowledge the trust risk to continue.');
      return;
    }
    guardrailHardBlock = false;
    guardrailAcknowledged = true;
    guardrailWarning = 'Override noted. Proceeding despite trust flags on this order.';
    guardrailLogId = recordGuardrailOverride({
      note: guardrailOverrideNote.trim(),
      item_hashes: $cart.map((item) => item.listing_hash),
      proof_states: proofLookup,
      risk_flags: Object.fromEntries(
        Object.entries(riskLookup).map(([key, risk]) => [key, risk?.flags?.[0]])
      ),
    });
    notifications.info('Override captured', 'Your acknowledgement is attached to this order.');
  }
</script>

<div class="checkout-page">
  <div class="container">
    {#if loading}
      <!-- Loading State -->
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading checkout...</p>
      </div>
    {:else if checkoutSuccess}
      <!-- Success State -->
      <div class="success-state">
        <span class="success-icon">✓</span>
        <h2>Order Placed Successfully!</h2>
        <p>Your transactions have been created on the Holochain DHT.</p>
        <div class="transaction-hashes">
          <h3>Transaction IDs:</h3>
          {#each transactionIds as id}
            <code class="transaction-hash">{id}</code>
          {/each}
        </div>
        <p class="redirect-message">Redirecting to transaction tracking...</p>
        <a href="/transactions?trust=overrides" class="btn btn-secondary">View trust audit now</a>
      </div>
    {:else if $cart.length === 0}
      <!-- Empty Cart -->
      <div class="error-state">
        <span class="error-icon">🛒</span>
        <h2>Your cart is empty</h2>
        <p>Add some items before checking out</p>
        <a href="/browse" class="btn btn-primary">Browse Marketplace</a>
      </div>
    {:else}
      <ConnectionNotice />
      <!-- Checkout Process -->
      <div class="checkout-header">
        <h1>Checkout</h1>
        <div class="steps-indicator">
          <div class="step" class:active={currentStep >= 1} class:completed={currentStep > 1}>
            <div class="step-number">1</div>
            <div class="step-label">Shipping</div>
          </div>
          <div class="step-divider"></div>
          <div class="step" class:active={currentStep >= 2} class:completed={currentStep > 2}>
            <div class="step-number">2</div>
            <div class="step-label">Payment</div>
          </div>
          <div class="step-divider"></div>
          <div class="step" class:active={currentStep >= 3}>
            <div class="step-number">3</div>
            <div class="step-label">Review</div>
          </div>
        </div>
      </div>

      <div class="checkout-content">
        <!-- Checkout Form -->
        <div class="checkout-form">
          {#if guardrailWarning}
            <div class="guardrail">
              <span class="guardrail-icon">⚠</span>
              <div class="guardrail-copy">
                <p class="guardrail-title">Review recommended</p>
                <p class="guardrail-text">{guardrailWarning}</p>
                {#if guardrailHardBlock}
                  <div class="guardrail-override">
                    <label for="guardrail-note">Type an acknowledgement to continue</label>
                    <input
                      id="guardrail-note"
                      type="text"
                      placeholder="I accept the trust risk for this order"
                      bind:value={guardrailOverrideNote}
                    />
                    <p class="guardrail-help">
                      We keep this note with the order for future trust/dispute review.
                    </p>
                  </div>
                {/if}
              </div>
              {#if guardrailHardBlock}
                <button class="btn btn-secondary" on:click={overrideGuardrail} disabled={!canOverride}>
                  Override & continue
                </button>
              {/if}
            </div>
          {/if}
          {#if currentStep === 1}
            <!-- Step 1: Shipping Address -->
            <div class="form-section">
              <h2>Shipping Address</h2>

              <div class="form-row">
                <div class="form-field">
                  <label for="name">Full Name *</label>
                  <input
                    id="name"
                    type="text"
                    bind:value={shippingAddress.name}
                    placeholder="John Doe"
                    required
                  />
                </div>

                <div class="form-field">
                  <label for="phone">Phone Number *</label>
                  <input
                    id="phone"
                    type="tel"
                    bind:value={shippingAddress.phone}
                    placeholder="(555) 123-4567"
                    required
                  />
                </div>
              </div>

              <div class="form-field">
                <label for="address_line_1">Address Line 1 *</label>
                <input
                  id="address_line_1"
                  type="text"
                  bind:value={shippingAddress.address_line_1}
                  placeholder="123 Main Street"
                  required
                />
              </div>

              <div class="form-field">
                <label for="address_line_2">Address Line 2</label>
                <input
                  id="address_line_2"
                  type="text"
                  bind:value={shippingAddress.address_line_2}
                  placeholder="Apt 4B"
                />
              </div>

              <div class="form-row">
                <div class="form-field">
                  <label for="city">City *</label>
                  <input
                    id="city"
                    type="text"
                    bind:value={shippingAddress.city}
                    placeholder="New York"
                    required
                  />
                </div>

                <div class="form-field">
                  <label for="state">State *</label>
                  <input
                    id="state"
                    type="text"
                    bind:value={shippingAddress.state}
                    placeholder="NY"
                    required
                  />
                </div>

                <div class="form-field">
                  <label for="postal_code">ZIP Code *</label>
                  <input
                    id="postal_code"
                    type="text"
                    bind:value={shippingAddress.postal_code}
                    placeholder="10001"
                    required
                  />
                </div>
              </div>

              <div class="form-field">
                <label for="country">Country *</label>
                <select id="country" bind:value={shippingAddress.country}>
                  <option value="United States">United States</option>
                  <option value="Canada">Canada</option>
                  <option value="United Kingdom">United Kingdom</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          {:else if currentStep === 2}
            <!-- Step 2: Payment Method -->
            <div class="form-section">
              <h2>Payment Method</h2>

              <div class="payment-methods">
                <button
                  class="payment-method-btn"
                  class:active={paymentMethod === 'crypto'}
                  on:click={() => (paymentMethod = 'crypto')}
                >
                  <span class="method-icon">₿</span>
                  <span class="method-name">Cryptocurrency</span>
                </button>
                <button
                  class="payment-method-btn"
                  class:active={paymentMethod === 'credit_card'}
                  on:click={() => (paymentMethod = 'credit_card')}
                >
                  <span class="method-icon">💳</span>
                  <span class="method-name">Credit/Debit Card</span>
                </button>
              </div>

              {#if paymentMethod === 'crypto'}
                <div class="payment-details">
                  <div class="form-field">
                    <label for="cryptoCurrency">Cryptocurrency</label>
                    <select id="cryptoCurrency" bind:value={paymentDetails.cryptoCurrency}>
                      <option value="ETH">Ethereum (ETH)</option>
                      <option value="BTC">Bitcoin (BTC)</option>
                      <option value="USDC">USD Coin (USDC)</option>
                      <option value="DAI">Dai (DAI)</option>
                    </select>
                  </div>

                  <div class="form-field">
                    <label for="walletAddress">Your Wallet Address *</label>
                    <input
                      id="walletAddress"
                      type="text"
                      bind:value={paymentDetails.walletAddress}
                      placeholder="0x..."
                      required
                    />
                    <p class="field-hint">
                      Payment will be escrowed until delivery confirmation
                    </p>
                  </div>
                </div>
              {:else}
                <div class="payment-details">
                  <div class="form-field">
                    <label for="cardNumber">Card Number *</label>
                    <input
                      id="cardNumber"
                      type="text"
                      bind:value={paymentDetails.cardNumber}
                      placeholder="1234 5678 9012 3456"
                      maxlength="19"
                      required
                    />
                  </div>

                  <div class="form-field">
                    <label for="cardName">Cardholder Name *</label>
                    <input
                      id="cardName"
                      type="text"
                      bind:value={paymentDetails.cardName}
                      placeholder="John Doe"
                      required
                    />
                  </div>

                  <div class="form-row">
                    <div class="form-field">
                      <label for="cardExpiry">Expiry Date *</label>
                      <input
                        id="cardExpiry"
                        type="text"
                        bind:value={paymentDetails.cardExpiry}
                        placeholder="MM/YY"
                        maxlength="5"
                        required
                      />
                    </div>

                    <div class="form-field">
                      <label for="cardCVV">CVV *</label>
                      <input
                        id="cardCVV"
                        type="text"
                        bind:value={paymentDetails.cardCVV}
                        placeholder="123"
                        maxlength="4"
                        required
                      />
                    </div>
                  </div>
                </div>
              {/if}
            </div>
          {:else if currentStep === 3}
            <!-- Step 3: Review Order -->
            <div class="form-section">
              <h2>Review Your Order</h2>

              <div class="review-section">
                <h3>Shipping Address</h3>
                <div class="review-address">
                  <p>{shippingAddress.name}</p>
                  <p>{shippingAddress.address_line_1}</p>
                  {#if shippingAddress.address_line_2}
                    <p>{shippingAddress.address_line_2}</p>
                  {/if}
                  <p>
                    {shippingAddress.city}, {shippingAddress.state} {shippingAddress.postal_code}
                  </p>
                  <p>{shippingAddress.country}</p>
                  <p>{shippingAddress.phone}</p>
                </div>
              </div>

              <div class="review-section">
                <h3>Payment Method</h3>
                <div class="review-payment">
                  {#if paymentMethod === 'crypto'}
                    <p>
                      <strong>Cryptocurrency:</strong> {paymentDetails.cryptoCurrency}
                    </p>
                    <p class="wallet-address">
                      Wallet: {paymentDetails.walletAddress.substring(0, 20)}...
                    </p>
                  {:else}
                    <p>
                      <strong>Card:</strong> •••• {paymentDetails.cardNumber.slice(-4)}
                    </p>
                    <p>{paymentDetails.cardName}</p>
                  {/if}
                </div>
              </div>

              <div class="review-section">
                <h3>Order Items</h3>
                <div class="review-items">
                  {#each $cart as item}
                    <div class="review-item">
                      <div class="item-thumbnail">
                        {#if item.photo_cid}
                          <img src={getIpfsUrl(item.photo_cid)} alt={item.title} />
                        {:else}
                          <div class="no-image">📷</div>
                        {/if}
                      </div>
                      <div class="item-details">
                        <p class="item-title">{item.title}</p>
                        <p class="item-quantity">Quantity: {item.quantity}</p>
                        <div class="item-trust">
                          <div class="trust-flags">
                            {#if proofLookup[item.listing_hash]}
                              <span class={`trust-pill proof ${proofLookup[item.listing_hash]}`}>
                                Proof {proofLookup[item.listing_hash]}
                              </span>
                            {:else}
                              <span class="trust-pill proof unknown">Proof unknown</span>
                            {/if}
                            {#if riskLookup[item.listing_hash]?.score > 0.5}
                              <span
                                class="trust-pill risk"
                                title={riskLookup[item.listing_hash]?.flags?.[0] || 'Potential anomaly detected'}
                              >
                                Risk {(riskLookup[item.listing_hash]?.score || 0).toFixed(2)}
                              </span>
                            {/if}
                          </div>
                        </div>
                      </div>
                      <div class="item-price">${(item.price * item.quantity).toFixed(2)}</div>
                    </div>
                    {#if riskLookup[item.listing_hash]}
                      <div class="insight-row">
                        <RiskInsightDrawer
                          risk={riskLookup[item.listing_hash]}
                          proof={proofLookup[item.listing_hash] || null}
                          open={false}
                        />
                      </div>
                    {/if}
                  {/each}
                </div>
              </div>
            </div>
          {/if}

          {#if error}
            <div class="alert alert-error">
              <span class="alert-icon">⚠️</span>
              {error}
            </div>
          {/if}

          {#if checkoutError}
            <div class="alert alert-error">
              <span class="alert-icon">⚠️</span>
              {checkoutError}
            </div>
          {/if}

          <!-- Navigation Buttons -->
          <div class="form-actions">
            {#if currentStep > 1}
              <button class="btn btn-secondary" on:click={previousStep}>
                ← Back
              </button>
            {/if}

            {#if currentStep < 3}
              <button class="btn btn-primary" on:click={nextStep} disabled={!$isConnected}>
                Continue →
              </button>
            {:else}
              <button
                class="btn btn-primary"
                on:click={completeCheckout}
                disabled={processingCheckout || !$isConnected}
              >
                {#if processingCheckout}
                  Processing Order...
                {:else}
                  Place Order
                {/if}
              </button>
            {/if}
          </div>
        </div>

        <!-- Order Summary Sidebar -->
        <div class="order-summary">
          <h2>Order Summary</h2>

          <div class="summary-items">
            {#each $cart as item}
              <div class="summary-item">
                <div class="item-meta">
                  <span class="item-name">{item.title} × {item.quantity}</span>
                  <div class="trust-flags">
                    {#if proofLookup[item.listing_hash]}
                      <span class={`trust-pill proof ${proofLookup[item.listing_hash]}`}>
                        Proof {proofLookup[item.listing_hash]}
                      </span>
                    {:else}
                      <span class="trust-pill proof unknown">Proof unknown</span>
                    {/if}
                    {#if riskLookup[item.listing_hash]?.score > 0.5}
                      <span
                        class="trust-pill risk"
                        title={riskLookup[item.listing_hash]?.flags?.[0] || 'Potential anomaly detected'}
                      >
                        Risk {(riskLookup[item.listing_hash]?.score || 0).toFixed(2)}
                      </span>
                    {/if}
                  </div>
                </div>
                <span class="item-price">${(item.price * item.quantity).toFixed(2)}</span>
              </div>
            {/each}
          </div>

          <div class="summary-divider"></div>

          <div class="summary-row">
            <span>Subtotal</span>
            <span>${$subtotal.toFixed(2)}</span>
          </div>

          <div class="summary-row">
            <span>Tax (8%)</span>
            <span>${$tax.toFixed(2)}</span>
          </div>

          <div class="summary-row">
            <span>Shipping</span>
            <span>${$shipping.toFixed(2)}</span>
          </div>

          <div class="summary-divider"></div>

          <div class="summary-total">
            <span>Total</span>
            <span>${$total.toFixed(2)}</span>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .checkout-page {
    min-height: 100vh;
    padding: 2rem 1rem;
    background: #f7fafc;
  }

  .container {
    max-width: 1200px;
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

  /* Success State */
  .success-state {
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

  .success-icon {
    font-size: 5rem;
    color: #38a169;
    margin-bottom: 1.5rem;
  }

  .success-state h2 {
    font-size: 2rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 1rem;
  }

  .success-state p {
    color: #718096;
    margin-bottom: 2rem;
  }

  .transaction-hashes {
    width: 100%;
    max-width: 600px;
    margin-bottom: 2rem;
  }

  .transaction-hashes h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1rem;
  }

  .transaction-hash {
    display: block;
    padding: 0.75rem;
    background: #f7fafc;
    border-radius: 0.375rem;
    font-family: monospace;
    font-size: 0.875rem;
    color: #4a5568;
    margin-bottom: 0.5rem;
    word-break: break-all;
  }

  .redirect-message {
    color: #4299e1;
    font-weight: 500;
  }

  /* Checkout Header */
  .checkout-header {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .checkout-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 2rem;
    text-align: center;
  }

  /* Steps Indicator */
  .steps-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
  }

  .step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  .step-number {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #e2e8f0;
    color: #718096;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    transition: all 0.3s;
  }

  .step.active .step-number {
    background: #4299e1;
    color: white;
  }

  .step.completed .step-number {
    background: #38a169;
    color: white;
  }

  .step-label {
    font-size: 0.875rem;
    color: #718096;
    font-weight: 500;
  }

  .step.active .step-label {
    color: #4299e1;
    font-weight: 600;
  }

  .step-divider {
    width: 60px;
    height: 2px;
    background: #e2e8f0;
  }

  /* Checkout Content */
  .checkout-content {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 2rem;
  }

  @media (max-width: 768px) {
    .checkout-content {
      grid-template-columns: 1fr;
    }
  }

  /* Checkout Form */
  .checkout-form {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .form-section {
    margin-bottom: 2rem;
  }

  .form-section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1.5rem;
  }

  .form-section h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1rem;
  }

  .form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .form-field {
    margin-bottom: 1rem;
  }

  .form-field label {
    display: block;
    font-size: 0.875rem;
    font-weight: 500;
    color: #4a5568;
    margin-bottom: 0.5rem;
  }

  .form-field input,
  .form-field select {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #cbd5e0;
    border-radius: 0.375rem;
    font-size: 1rem;
    transition: border-color 0.2s;
  }

  .form-field input:focus,
  .form-field select:focus {
    outline: none;
    border-color: #4299e1;
  }

  .field-hint {
    margin-top: 0.5rem;
    font-size: 0.75rem;
    color: #718096;
  }

  /* Payment Methods */
  .payment-methods {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .payment-method-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 1.5rem;
    border: 2px solid #e2e8f0;
    border-radius: 0.375rem;
    background: white;
    cursor: pointer;
    transition: all 0.2s;
  }

  .payment-method-btn:hover {
    border-color: #4299e1;
  }

  .payment-method-btn.active {
    border-color: #4299e1;
    background: #ebf8ff;
  }

  .method-icon {
    font-size: 2rem;
  }

  .method-name {
    font-weight: 600;
    color: #2d3748;
  }

  .payment-details {
    padding: 1.5rem;
    background: #f7fafc;
    border-radius: 0.375rem;
  }

  /* Review Sections */
  .review-section {
    padding: 1.5rem;
    background: #f7fafc;
    border-radius: 0.375rem;
    margin-bottom: 1.5rem;
  }

  .review-address p,
  .review-payment p {
    color: #4a5568;
    line-height: 1.6;
  }

  .wallet-address {
    font-family: monospace;
    font-size: 0.875rem;
  }

  .review-items {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .review-item {
    display: grid;
    grid-template-columns: 60px 1fr auto;
    gap: 1rem;
    align-items: center;
  }

  .item-thumbnail {
    width: 60px;
    height: 60px;
    border-radius: 0.375rem;
    overflow: hidden;
    background: white;
  }

  .item-thumbnail img {
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
    font-size: 1.5rem;
    color: #cbd5e0;
  }

  .item-details {
    flex: 1;
  }

  .item-title {
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.25rem;
  }

  .item-quantity {
    font-size: 0.875rem;
    color: #718096;
  }

  .item-trust {
    margin-top: 0.35rem;
  }

  .item-price {
    font-size: 1.125rem;
    font-weight: 700;
    color: #38a169;
  }

  /* Order Summary */
  .order-summary {
    background: white;
    border-radius: 0.5rem;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    height: fit-content;
    position: sticky;
    top: 2rem;
  }

  .order-summary h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1.5rem;
  }

  .summary-items {
    margin-bottom: 1rem;
  }

  .summary-item {
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
  }

  .item-meta {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    flex: 1;
  }

  .item-name {
    flex: 1;
    color: #4a5568;
  }

  .trust-flags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }

  .trust-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    border: 1px solid #e2e8f0;
    font-size: 0.75rem;
    font-weight: 700;
    color: #2d3748;
    background: #edf2f7;
  }

  .trust-pill.proof.fulfilled {
    background: #dcfce7;
    border-color: #86efac;
    color: #166534;
  }

  .trust-pill.proof.pending,
  .trust-pill.proof.requested {
    background: #fef3c7;
    border-color: #fcd34d;
    color: #92400e;
  }

  .trust-pill.proof.denied {
    background: #fee2e2;
    border-color: #fca5a5;
    color: #991b1b;
  }

  .trust-pill.proof.unknown {
    background: #e2e8f0;
    border-color: #cbd5e0;
    color: #4a5568;
  }

  .trust-pill.risk {
    background: #fff7ed;
    border-color: #fdba74;
    color: #9a3412;
  }

  .summary-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    color: #4a5568;
  }

  .summary-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 1rem 0;
  }

  .summary-total {
    display: flex;
    justify-content: space-between;
    font-size: 1.25rem;
    font-weight: 700;
    color: #2d3748;
  }

  .summary-total span:last-child {
    color: #38a169;
  }

  /* Form Actions */
  .form-actions {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 2rem;
  }

  /* Buttons */
  .btn {
    padding: 0.75rem 2rem;
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
    flex: 1;
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

  /* Alerts */
  .alert {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    border-radius: 0.375rem;
    margin-bottom: 1rem;
  }

  .alert-icon {
    font-size: 1.25rem;
  }

  .alert-error {
    background: #fed7d7;
    border: 1px solid #fc8181;
    color: #742a2a;
  }

  /* Trust Guardrail */
  .guardrail {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    margin-bottom: 1.5rem;
    background: #fffaf0;
    border: 1px solid #fbd38d;
    border-radius: 0.5rem;
    color: #744210;
  }

  .guardrail-icon {
    font-size: 1.5rem;
  }

  .guardrail-copy {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    flex: 1;
  }

  .guardrail-title {
    font-weight: 700;
    margin-bottom: 0.15rem;
  }

  .guardrail-text {
    color: #744210;
    margin: 0;
  }

  .guardrail .btn {
    margin-left: auto;
    white-space: nowrap;
  }

  .guardrail-override {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .guardrail-override label {
    font-weight: 600;
    font-size: 0.9rem;
    color: #744210;
  }

  .guardrail-override input {
    padding: 0.6rem 0.75rem;
    border: 1px solid #f6ad55;
    border-radius: 0.4rem;
    background: #fffaf0;
    color: #744210;
  }

  .guardrail-override input:focus {
    outline: none;
    border-color: #dd6b20;
    box-shadow: 0 0 0 3px rgba(221, 107, 32, 0.2);
  }

  .guardrail-help {
    margin: 0;
    font-size: 0.8rem;
    color: #9a6b1a;
  }
</style>
