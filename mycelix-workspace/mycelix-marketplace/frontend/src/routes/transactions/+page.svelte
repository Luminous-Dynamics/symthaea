<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  /**
   * Transaction Tracking Page
   *
   * View and manage all transactions with:
   * - Transaction list (purchases and sales)
   * - Status tracking (pending, shipped, delivered, completed)
   * - Transaction timeline
   * - Actions (confirm delivery, leave review, file dispute)
   * - Shipping information
   * - Listing details with IPFS photos
   *
   * This demonstrates:
   * - Transaction lifecycle management
   * - Status updates and tracking
   * - Buyer and seller actions
   * - Integration with reviews and disputes
   */

  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { getIpfsUrl } from '$lib/ipfs/ipfsClient';
  import TrustBadge from '$lib/components/TrustBadge.svelte';
  import RiskChip from '$lib/components/RiskChip.svelte';
  import ConnectionNotice from '$lib/components/ConnectionNotice.svelte';
  import { initHolochainClient } from '$lib/holochain';
  import { getMyPurchases, getMySales, confirmDelivery as confirmDeliveryZome, markAsShipped as markAsShippedZome } from '$lib/holochain/transactions';
  import { notifications, isConnected, getProofStatus } from '$lib/stores';
  import { requestReputationProof } from '$lib/reputation';
  import { analyzeRisk } from '$lib/risk';
  import type { RiskSignal } from '$types';
  import { guardrailOverrides, type GuardrailOverrideEntry } from '$lib/stores/guardrails';
  import { getMockTransactions } from '$lib/mock/transactions';
  import type { Transaction, TransactionStatus } from '$types';
  import RiskInsightDrawer from '$lib/components/RiskInsightDrawer.svelte';
  import { generateQRDataURL } from '$lib/utils/qr';

// Extended transaction type with UI-specific fields
  interface TransactionWithUI extends Transaction {
    type: 'purchase' | 'sale';
    other_party_name?: string;
    other_party_trust_score?: number;
    can_confirm_delivery?: boolean;
    can_leave_review?: boolean;
    can_file_dispute?: boolean;
    can_mark_shipped?: boolean;
    can_cancel?: boolean;
  }

  // Transactions state
  let transactions: TransactionWithUI[] = [];
  let loading = true;
  let error = '';
  let usingMockData = false;
  const gateways = ['https://ipfs.io/ipfs/', 'https://cloudflare-ipfs.com/ipfs/'];
  let lastUpdated = 0;
  $: lastUpdatedLabel = lastUpdated
    ? new Date(lastUpdated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';
  const TRUST_DEEP_LINK = '#trust';

  // Filter state
  let filterType: 'all' | 'purchases' | 'sales' = 'all';
  let filterStatus: TransactionStatus | 'all' = 'all';
  let filterTrust: boolean = false;

  // Selected transaction for detail view
  let selectedTransaction: TransactionWithUI | null = null;
  let selectedProofStatus: string | null = null;
  let selectedAudits: GuardrailOverrideEntry[] = [];

  // Action state
  let actionInProgress = false;
  let actionError = '';
  let actionSuccess = '';
  let riskLookup: Record<string, RiskSignal> = {};
  let riskCache: Record<string, RiskSignal> = loadRiskCache();
  let qrCache: Record<string, string> = {};

  /**
   * Load transactions from Holochain
   */
  onMount(async () => {
    if (typeof window !== 'undefined') {
      const params = new URL(window.location.href).searchParams;
      if (params.get('trust') === 'overrides') {
        filterTrust = true;
      }
    }
    await loadTransactions();
  });

  async function loadTransactions() {
    loading = true;
    error = '';

    try {
      // Initialize Holochain client
      const client = await initHolochainClient();

      // Fetch purchases and sales in parallel
      const [purchases, sales] = await Promise.all([
        getMyPurchases(client),
        getMySales(client),
      ]);

      // TODO: Enhance with listing details and seller/buyer info
      // For now, we'll use basic transaction data
      const purchasesWithType: TransactionWithUI[] = purchases.map((t) => ({
        ...t,
        type: 'purchase' as const,
        can_confirm_delivery: t.status === 'shipped',
        can_leave_review: t.status === 'delivered' || t.status === 'completed',
        can_file_dispute: t.status === 'shipped' || t.status === 'delivered',
      }));

      const salesWithType: TransactionWithUI[] = sales.map((t) => ({
        ...t,
        type: 'sale' as const,
        can_mark_shipped: t.status === 'pending',
        can_cancel: t.status === 'pending',
      }));

      // Combine and sort by creation date (newest first)
      transactions = [...purchasesWithType, ...salesWithType].sort(
        (a, b) => b.created_at - a.created_at
      );

      notifications.success('Transactions loaded', `Found ${transactions.length} transactions`);
      riskLookup = await analyzeRisk(toListingLike(transactions));
      persistRiskCache(riskLookup);
      lastUpdated = Date.now();
    } catch (e: any) {
      const mockTx = getMockTransactions().map((t) => ({
        ...t,
        type: t.buyer_agent_id === 'mock-agent' ? ('sale' as const) : ('purchase' as const),
        can_confirm_delivery: t.status === 'shipped',
        can_leave_review: t.status === 'delivered' || t.status === 'completed',
        can_file_dispute: t.status === 'shipped' || t.status === 'delivered',
        can_mark_shipped: t.status === 'pending',
        can_cancel: t.status === 'pending',
      }));

      transactions = mockTx.sort((a, b) => b.created_at - a.created_at);
      usingMockData = true;
      error = e.message || 'Using offline preview data';
      notifications.warning('Offline Preview', error);
      riskLookup = await analyzeRisk(toListingLike(transactions));
      persistRiskCache(riskLookup);
      lastUpdated = Date.now();
    } finally {
      loading = false;
    }
  }

  /**
   * Filter transactions
   */
  $: filteredTransactions = transactions.filter((t) => {
    if (filterType !== 'all' && t.type !== filterType.slice(0, -1)) return false;
    if (filterStatus !== 'all' && t.status !== filterStatus) return false;
    if (filterTrust && guardrailsFor(t).length === 0) return false;
    return true;
  });

  /**
   * Select transaction for detail view
   */
  function selectTransaction(transaction: TransactionWithUI) {
    selectedTransaction = transaction;
    actionError = '';
    actionSuccess = '';
    selectedProofStatus = proofStatusFor(transaction);
    selectedAudits = guardrailsFor(transaction);
  }

  function proofStatusFor(transaction: TransactionWithUI): string | null {
    return getProofStatus(transaction.seller_agent_id, transaction.listing_hash || undefined);
  }

  function persistRiskCache(risk: Record<string, RiskSignal>) {
    try {
      localStorage.setItem('mycelix_risk_cache_tx', JSON.stringify(risk));
      riskCache = risk;
    } catch {
      // ignore
    }
  }

  function loadRiskCache(): Record<string, RiskSignal> {
    try {
      const raw = localStorage.getItem('mycelix_risk_cache_tx');
      return raw ? (JSON.parse(raw) as Record<string, RiskSignal>) : {};
    } catch {
      return {};
    }
  }

  function riskFor(transaction: TransactionWithUI): RiskSignal | null {
    const key = transaction.listing_hash || transaction.id;
    return riskLookup[key] || riskCache[key] || null;
  }

  function riskChanged(transaction: TransactionWithUI): boolean {
    const key = transaction.listing_hash || transaction.id;
    const cached = riskCache[key]?.score;
    const current = riskLookup[key]?.score;
    const snapshot = transaction.risk_snapshot_hash;
    if (snapshot && riskLookup[key]) {
      const currentHash = btoa(unescape(encodeURIComponent(JSON.stringify(riskLookup[key])))).slice(
        0,
        snapshot.length
      );
      if (currentHash !== snapshot) return true;
    }
    if (cached === undefined || current === undefined) return false;
    return Math.abs((current || 0) - (cached || 0)) > 0.15;
  }

  function riskDelta(transaction: TransactionWithUI): number | null {
    const key = transaction.listing_hash || transaction.id;
    if (riskLookup[key]?.score === undefined || riskCache[key]?.score === undefined) return null;
    return (riskLookup[key]?.score || 0) - (riskCache[key]?.score || 0);
  }

  function exportAudit(tx: TransactionWithUI) {
    const data = {
      transaction_id: tx.id,
      listing_hash: tx.listing_hash,
      risk_snapshot_hash: tx.risk_snapshot_hash || null,
      current_risk: riskFor(tx),
      risk_delta: riskFor(tx)?.score && riskCache[tx.listing_hash || tx.id]?.score
        ? (riskFor(tx)?.score || 0) - (riskCache[tx.listing_hash || tx.id]?.score || 0)
        : null,
      proof_status: proofStatusFor(tx),
      audits: guardrailsFor(tx),
      risk_changed: riskChanged(tx),
      seller: {
        id: tx.seller_agent_id,
        name: tx.seller_name,
      },
      buyer: {
        id: tx.buyer_agent_id,
        name: tx.buyer_name,
      },
      link: `${window.location.origin}/transactions#${tx.id}`,
      short_link: `/transactions#${tx.id.slice(0, 10)}`,
      link_type: 'short+full',
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const snap = tx.risk_snapshot_hash ? tx.risk_snapshot_hash.slice(0, 6) : 'nosnap';
    const delta =
      riskFor(tx)?.score && riskCache[tx.listing_hash || tx.id]?.score
        ? ((riskFor(tx)?.score || 0) - (riskCache[tx.listing_hash || tx.id]?.score || 0)).toFixed(2)
        : 'na';
    a.href = url;
    a.download = `trust-audit-${snap}-d${delta}-${tx.id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    notifications.success('Audit exported', 'Downloaded trust audit JSON.');
  }

  function qrFor(tx: TransactionWithUI): string {
    if (qrCache[tx.id]) return qrCache[tx.id];
    if (typeof document === 'undefined') return '';
    const url = `${window.location.origin}/transactions#${tx.id}`;
    const dataUrl = generateQRDataURL(url, 120);
    qrCache[tx.id] = dataUrl;
    return dataUrl;
  }

  function toListingLike(tx: TransactionWithUI[]): any[] {
    return tx.map((t) => ({
      id: t.listing_hash || t.id,
      listing_hash: t.listing_hash,
      title: t.listing_title,
      description: t.listing_title,
      price: t.unit_price,
      category: 'Other',
      photos_ipfs_cids: t.listing_photo_cid ? [t.listing_photo_cid] : [],
      seller_agent_id: t.seller_agent_id,
      created_at: t.created_at,
      status: 'active',
    }));
  }

  async function requestProofFor(tx: TransactionWithUI) {
    if (!tx.seller_agent_id) return;
    try {
      await requestReputationProof(tx.seller_agent_id, tx.listing_hash);
      notifications.success('Proof requested', 'Seller notified to publish proof.');
    } catch (e: any) {
      notifications.error(e?.message || 'Unable to request proof right now.');
    }
  }

  /**
   * Open review form for a transaction
   */
  function goToReview(transaction: TransactionWithUI | null) {
    if (!transaction || !$isConnected) return;
    const params = new URLSearchParams({
      transaction: transaction.id,
      listing: transaction.listing_hash,
      title: transaction.listing_title,
      seller: transaction.seller_name,
    });
    goto(`/submit-review?${params.toString()}`);
  }

  /**
   * Open dispute form for a transaction
   */
  function goToDispute(transaction: TransactionWithUI | null) {
    if (!transaction || !$isConnected) return;
    const params = new URLSearchParams({
      transaction: transaction.id,
      title: transaction.listing_title,
      seller: transaction.seller_name,
    });
    goto(`/file-dispute?${params.toString()}`);
  }

  /**
   * Confirm delivery (buyer action)
   */
  async function confirmDelivery() {
    if (!selectedTransaction || !$isConnected) return;

    actionInProgress = true;
    actionError = '';
    actionSuccess = '';

    try {
      // Initialize Holochain client
      const client = await initHolochainClient();

      // Confirm delivery via zome call
      const updatedTransaction = await confirmDeliveryZome(client, selectedTransaction.id);

      // Update local state
      selectedTransaction = {
        ...selectedTransaction,
        ...updatedTransaction,
        can_confirm_delivery: false,
        can_leave_review: true,
      };

      // Update transactions list
      transactions = transactions.map((t) =>
        t.id === selectedTransaction?.id ? selectedTransaction : t
      );

      actionSuccess = 'Delivery confirmed successfully!';
      notifications.success('Delivery confirmed', 'You can now leave a review');
    } catch (e: any) {
      actionError = e.message || 'Failed to confirm delivery';
      notifications.error('Confirmation failed', actionError);
      console.error('Error confirming delivery:', e);
    } finally {
      actionInProgress = false;
    }
  }

  /**
   * Mark as shipped (seller action)
   */
  async function markAsShipped() {
    if (!selectedTransaction || !$isConnected) return;

    const trackingNumber = prompt('Enter tracking number:');
    if (!trackingNumber || !trackingNumber.trim()) {
      notifications.warning('Cancelled', 'Tracking number is required');
      return;
    }

    actionInProgress = true;
    actionError = '';
    actionSuccess = '';

    try {
      // Initialize Holochain client
      const client = await initHolochainClient();

      // Mark as shipped via zome call
      const updatedTransaction = await markAsShippedZome(
        client,
        selectedTransaction.id,
        trackingNumber.trim()
      );

      // Update local state
      selectedTransaction = {
        ...selectedTransaction,
        ...updatedTransaction,
        can_mark_shipped: false,
      };

      // Update transactions list
      transactions = transactions.map((t) =>
        t.id === selectedTransaction?.id ? selectedTransaction : t
      );

      actionSuccess = 'Marked as shipped successfully!';
      notifications.success('Shipped', `Tracking: ${trackingNumber}`);
    } catch (e: any) {
      actionError = e.message || 'Failed to mark as shipped';
      notifications.error('Shipping update failed', actionError);
      console.error('Error marking as shipped:', e);
    } finally {
      actionInProgress = false;
    }
  }

  /**
   * Format date
   */
  function formatDate(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  }

  /**
   * Format date with time
   */
  function formatDateTime(timestamp: number): string {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  }

  function openTrustAudit() {
    filterTrust = true;
    selectedTransaction = null;
  }

  /**
   * Get status badge class
   */
  function getStatusClass(status: string): string {
    const statusMap: Record<string, string> = {
      pending: 'status-warning',
      shipped: 'status-info',
      delivered: 'status-success',
      completed: 'status-success',
      cancelled: 'status-error',
      disputed: 'status-error',
    };
    return statusMap[status] || 'status-default';
  }

  /**
   * Get status display name
   */
  function getStatusName(status: string): string {
    return status.charAt(0).toUpperCase() + status.slice(1);
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

  function guardrailsFor(transaction: TransactionWithUI): GuardrailOverrideEntry[] {
    const idMatches = (entry: GuardrailOverrideEntry) =>
      entry.transaction_ids?.includes(transaction.id);
    const listingMatches = (entry: GuardrailOverrideEntry) =>
      !!transaction.listing_hash && entry.item_hashes?.includes(transaction.listing_hash);
    return $guardrailOverrides.filter((entry) => idMatches(entry) || listingMatches(entry));
  }

  $: if (selectedTransaction) {
    selectedAudits = guardrailsFor(selectedTransaction);
  }
</script>

<div class="transactions-page">
  <div class="container">
    {#if loading}
      <!-- Loading State -->
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading transactions...</p>
      </div>
    {:else if error}
      <!-- Error State -->
      <div class="error-state">
        <span class="error-icon">⚠️</span>
        <p>{error}</p>
      </div>
    {:else}
      <ConnectionNotice />
      {#if usingMockData}
        <div class="mock-badge">Offline preview data</div>
      {/if}
      <!-- Page Header -->
      <div class="page-header">
        <div>
          <h1>My Transactions</h1>
          <p>Track your purchases and sales</p>
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
          <button class="btn btn-secondary" on:click={openTrustAudit} title="Show only trust overrides">
            Trust audit
          </button>
          {#if filterTrust}
            <button class="btn btn-tertiary" on:click={() => (filterTrust = false)}>
              Clear trust filter
            </button>
          {/if}
          <button class="btn btn-tertiary" on:click={loadTransactions} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <!-- Filters -->
      <div class="filters-bar">
        <div class="filter-group">
          <label for="filter-type">Type:</label>
          <select id="filter-type" bind:value={filterType}>
            <option value="all">All Transactions</option>
            <option value="purchases">Purchases</option>
            <option value="sales">Sales</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="filter-status">Status:</label>
          <select id="filter-status" bind:value={filterStatus}>
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="shipped">Shipped</option>
            <option value="delivered">Delivered</option>
            <option value="completed">Completed</option>
          </select>
        </div>

        <div class="filter-group trust-toggle">
          <label>
            <input
              type="checkbox"
              bind:checked={filterTrust}
              aria-label="Show only transactions with trust overrides"
            />
            Trust overrides only
          </label>
          <span class="trust-count">{$guardrailOverrides.length} logged</span>
        </div>

        <div class="results-count">
          {filteredTransactions.length} transaction{filteredTransactions.length !== 1 ? 's' : ''}
        </div>
      </div>

      <!-- Transactions List -->
      {#if filteredTransactions.length === 0}
        <div class="empty-state">
          <span>📦</span>
          <p>No transactions found</p>
          <a href="/browse" class="btn btn-primary">Browse Marketplace</a>
        </div>
      {:else}
        {#if filterTrust}
          <div class="filter-banner">
            <span>Showing trust overrides only.</span>
            <button class="btn btn-tertiary" on:click={() => (filterTrust = false)}>
              Clear filter
            </button>
          </div>
        {/if}
        <div class="transactions-list">
          {#each filteredTransactions as transaction}
            <button
              class="transaction-card"
              class:selected={selectedTransaction?.transaction_hash ===
                transaction.transaction_hash}
              on:click={() => selectTransaction(transaction)}
              on:keydown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  selectTransaction(transaction);
                }
              }}
              aria-label="View transaction {transaction.id.slice(0, 8)}"
            >
              <!-- Transaction Header -->
              <div class="transaction-header">
                <div class="transaction-type">
                  {#if transaction.type === 'purchase'}
                    <span class="type-badge type-purchase">Purchase</span>
                  {:else}
                    <span class="type-badge type-sale">Sale</span>
                  {/if}
                  {#if usingMockData}
                    <span class="preview-pill">Preview</span>
                  {/if}
                  {#if guardrailsFor(transaction).length}
                    {#each guardrailsFor(transaction).slice(0, 1) as audit}
                      <span
                        class="override-pill"
                        title={`Trust override: ${audit.note}${Object.values(audit.risk_flags || {}).filter(Boolean)[0] ? ` · ${Object.values(audit.risk_flags || {}).filter(Boolean)[0]}` : ''}`}
                      >
                        Trust override
                      </span>
                    {/each}
                  {/if}
                </div>

                <span class={`status-badge ${getStatusClass(transaction.status)}`}>
                  {getStatusName(transaction.status)}
                </span>
                <div class="card-actions">
                  <button
                    class="btn btn-tertiary small"
                    type="button"
                    on:click|stopPropagation={() => requestProofFor(transaction)}
                    title="Request proof from seller"
                  >
                    Request proof
                  </button>
                </div>
              </div>

              <!-- Transaction Content -->
              <div class="transaction-content">
              <div class="transaction-thumbnail">
                {#if transaction.listing_photo_cid}
                  <img
                    src={getIpfsUrl(transaction.listing_photo_cid)}
                    alt={transaction.listing_title}
                    data-gw="0"
                    on:error={(e) => handleImageError(e, transaction.listing_photo_cid)}
                  />
                {:else}
                  <div class="no-image">📷</div>
                {/if}
              </div>

                <div class="transaction-info">
                  <h3>{transaction.listing_title}</h3>
                  <p class="transaction-date">{formatDate(transaction.created_at)}</p>

                  {#if transaction.type === 'purchase'}
                    <p class="transaction-party">
                      Seller: <strong>{transaction.seller_name}</strong>
                      {#if proofStatusFor(transaction)}
                        <span
                          class={`proof-pill ${proofStatusFor(transaction)}`}
                          title={`Proof ${proofStatusFor(transaction)}`}
                        >
                          {proofStatusFor(transaction)}
                        </span>
                        <button
                          class="request-link"
                          type="button"
                          on:click|stopPropagation={() => requestProofFor(transaction)}
                          title="Ask seller to attach proof"
                        >
                          Request proof
                        </button>
                      {/if}
                    {#if riskFor(transaction) && (riskFor(transaction)?.score || 0) > 0.5}
                      <span
                        class="risk-pill"
                        title={riskFor(transaction)?.flags?.[0] || 'Potential anomaly detected'}
                      >
                        Review suggested
                      </span>
                      {#if riskChanged(transaction)}
                        <span
                          class="risk-change"
                          title={`Risk changed since last snapshot${riskDelta(transaction) !== null ? ` (${riskDelta(transaction)?.toFixed(2)})` : ''}`}
                        >
                          Risk changed
                        </span>
                      {/if}
                    {/if}
                  </p>
                  {:else}
                    <p class="transaction-party">
                      Buyer: <strong>{transaction.buyer_name}</strong>
                    </p>
                  {/if}

                  <p class="transaction-price">${transaction.total_price.toFixed(2)}</p>
                  <div class="inline-insight">
                    <RiskInsightDrawer
                      risk={riskFor(transaction)}
                      proof={proofStatusFor(transaction)}
                      open={false}
                    />
                  </div>
                  {#if guardrailsFor(transaction).length}
                    {#each guardrailsFor(transaction).slice(0, 1) as audit}
                      <div class="override-summary">
                        <span class="override-pill">Trust override</span>
                        <p class="override-text">
                          “{audit.note}”
                          {#if Object.values(audit.risk_flags || {}).some(Boolean)}
                            · {Object.values(audit.risk_flags || {}).filter(Boolean)[0]}
                          {/if}
                        </p>
                      </div>
                    {/each}
                  {/if}
                </div>
              </div>
            </button>
          {/each}
        </div>
      {/if}
    {/if}

    <!-- Transaction Detail Modal -->
    {#if selectedTransaction}
      {@const tx = selectedTransaction}
      <div
        class="transaction-modal"
        on:click={(event) => {
          if (event.currentTarget === event.target) {
            selectedTransaction = null;
          }
        }}
        on:keydown={(e) => {
          if (e.key === 'Escape') {
            selectedTransaction = null;
          }
        }}
        tabindex="-1"
        role="presentation"
      >
        <div
          class="modal-content"
          role="dialog"
          aria-modal="true"
          aria-labelledby="transaction-modal-title"
        >
          <button class="modal-close" on:click={() => (selectedTransaction = null)}>
            ✕
          </button>

          <h2 id="transaction-modal-title">Transaction Details</h2>

          <!-- Transaction Overview -->
          <div class="modal-section">
            <div class="section-header">
              <div class="title-row">
                <h3>{tx.listing_title}</h3>
                {#if usingMockData}
                  <span class="preview-pill">Preview</span>
                {/if}
              </div>
              <span class={`status-badge ${getStatusClass(tx.status)}`}>
                {getStatusName(tx.status)}
              </span>
            </div>
            <div class="risk-header">
              <RiskChip risk={riskFor(tx)} expanded={true} />
              {#if riskFor(tx) && (riskFor(tx)?.score || 0) > 0.5}
                <button class="btn btn-secondary" on:click={() => requestProofFor(tx)}>
                  Request proof
                </button>
              {/if}
            </div>

            {#if tx.listing_photo_cid}
              <img
                src={getIpfsUrl(tx.listing_photo_cid)}
                alt={tx.listing_title}
                class="detail-image"
                data-gw="0"
                on:error={(e) => handleImageError(e, tx.listing_photo_cid)}
              />
            {/if}

            <div class="transaction-details">
              <div class="detail-row">
                <span class="detail-label">Transaction ID:</span>
                <code class="detail-value">
                  {tx.id.substring(0, 30)}...
                </code>
              </div>

              <div class="detail-row">
                <span class="detail-label">Type:</span>
                <span class="detail-value">
                  {tx.type === 'purchase' ? 'Purchase' : 'Sale'}
                </span>
              </div>

              <div class="detail-row">
                <span class="detail-label">Total Price:</span>
                <span class="detail-value">${tx.total_price.toFixed(2)}</span>
              </div>

              <div class="detail-row">
                <span class="detail-label">Quantity:</span>
                <span class="detail-value">{tx.quantity}</span>
              </div>

              {#if tx.type === 'purchase'}
                <div class="detail-row">
                  <span class="detail-label">Seller:</span>
                  <div class="detail-value">
                    <span>{tx.seller_name}</span>
                    <TrustBadge
                      trustScore={tx.seller_trust_score}
                      size="small"
                      showLabel={false}
                      clickable={true}
                      agentId={tx.seller_agent_id}
                      on:click={() => goto(`/listing/${tx.listing_hash}${TRUST_DEEP_LINK}`)}
                    />
                    {#if selectedProofStatus}
                      <span class={`proof-pill ${selectedProofStatus}`} title={`Proof ${selectedProofStatus}`}>
                        {selectedProofStatus}
                      </span>
                    {/if}
                  </div>
                </div>
              {:else}
                <div class="detail-row">
                  <span class="detail-label">Buyer:</span>
                  <div class="detail-value">
                    <span>{tx.buyer_name}</span>
                    {#if tx.buyer_trust_score}
                      <TrustBadge
                        trustScore={tx.buyer_trust_score}
                        size="small"
                        showLabel={false}
                        clickable={true}
                        agentId={tx.buyer_agent_id}
                        on:click={() => goto(`/listing/${tx.listing_hash}${TRUST_DEEP_LINK}`)}
                      />
                    {/if}
                  </div>
                </div>
              {/if}
            </div>
          </div>

          <!-- Timeline -->
          <div class="modal-section">
            <h3>Transaction Timeline</h3>

            <div class="timeline">
                <div class="timeline-item completed">
                  <div class="timeline-icon">✓</div>
                  <div class="timeline-content">
                    <p class="timeline-title">Order Placed</p>
                    <p class="timeline-date">{formatDateTime(tx.created_at)}</p>
                  </div>
                </div>

                {#if tx.shipped_at}
                  <div class="timeline-item completed">
                    <div class="timeline-icon">✓</div>
                    <div class="timeline-content">
                      <p class="timeline-title">Shipped</p>
                      <p class="timeline-date">
                        {formatDateTime(tx.shipped_at)}
                      </p>
                      {#if tx.tracking_number}
                        <p class="tracking-number">
                          Tracking: {tx.tracking_number}
                        </p>
                      {/if}
                    </div>
                  </div>
                {/if}

                {#if tx.delivered_at}
                  <div class="timeline-item completed">
                    <div class="timeline-icon">✓</div>
                    <div class="timeline-content">
                      <p class="timeline-title">Delivered</p>
                      <p class="timeline-date">
                        {formatDateTime(tx.delivered_at)}
                      </p>
                    </div>
                  </div>
                {/if}

                {#if tx.completed_at}
                  <div class="timeline-item completed">
                    <div class="timeline-icon">✓</div>
                    <div class="timeline-content">
                      <p class="timeline-title">Completed</p>
                      <p class="timeline-date">
                        {formatDateTime(tx.completed_at)}
                      </p>
                    </div>
                  </div>
                {/if}
              </div>
          </div>

          <!-- Trust Audit Trail -->
          <div class="modal-section audit-print">
            <div class="section-header">
              <h3>Trust Audit Trail</h3>
              <div class="card-actions">
                {#if tx.risk_snapshot_hash}
                  <span class="snapshot-pill" title="Risk snapshot recorded at checkout">
                    Snapshot: {tx.risk_snapshot_hash.slice(0, 8)}...
                  </span>
                {/if}
                        {#if riskChanged(tx)}
                          <span class="risk-change" title="Risk changed since checkout">Risk changed</span>
                        {/if}
                {#if selectedAudits.length}
                  <button class="btn btn-tertiary" on:click={() => exportAudit(tx)}>
                    Export audit
                  </button>
                  <button class="btn btn-tertiary" on:click={() => window.print()}>
                    Print
                  </button>
                {/if}
              </div>
            </div>
            <div class="audit-print-header">
              <p class="audit-label">Transaction</p>
              <p class="audit-note">{tx.id}</p>
              <p class="audit-label">Seller</p>
              <p class="audit-note">{tx.seller_name} ({tx.seller_agent_id})</p>
              <p class="audit-label">Buyer</p>
              <p class="audit-note">{tx.buyer_name} ({tx.buyer_agent_id})</p>
              <p class="audit-label">Proof</p>
              <p class="audit-note">{selectedProofStatus || 'Unknown'}</p>
              <p class="audit-label">Risk delta</p>
              <p class="audit-note">
                {(riskFor(tx)?.score || 0).toFixed(2)}
                {#if riskCache[tx.listing_hash || tx.id]?.score !== undefined}
                  → {(riskCache[tx.listing_hash || tx.id]?.score || 0).toFixed(2)}
                  ({((riskFor(tx)?.score || 0) - (riskCache[tx.listing_hash || tx.id]?.score || 0)).toFixed(2)})
                {/if}
              </p>
              <p class="audit-label">Link</p>
              <p class="audit-note">
                <span class="audit-link">{window.location.origin}/transactions#{tx.id}</span>
                <span class="audit-link">/transactions#{tx.id.slice(0, 10)}…</span>
                <span class="qr-box" aria-label="Scan to open transaction">
                  <span class="qr-text">{tx.id.slice(0, 12)}…</span>
                  <span class="qr-hint">Scan</span>
                </span>
                <span class="qr-box" aria-label="Short link fallback">
                  <span class="qr-text">/tx/{tx.id.slice(0, 10)}</span>
                  <span class="qr-hint">Fallback</span>
                </span>
                {#if typeof window !== 'undefined' && qrFor(tx)}
                  <img class="qr-image" src={qrFor(tx)} alt="QR to transaction" loading="lazy" />
                {:else}
                  <span class="qr-fallback">Scan unavailable · use short link</span>
                {/if}
              </p>
            </div>
            {#if selectedAudits.length === 0}
              <p class="audit-empty">No guardrail overrides recorded for this transaction.</p>
            {:else}
              <div class="audit-list">
                {#each selectedAudits as audit (audit.id)}
                  <div class="audit-card">
                    <div class="audit-header">
                      <div>
                        <p class="audit-label">Override note</p>
                        <p class="audit-note">“{audit.note}”</p>
                        {#if riskChanged(tx)}
                          <span class="risk-change" title="Risk changed since checkout (current vs snapshot)">
                            Risk changed
                          </span>
                        {/if}
                        {#if selectedProofStatus}
                          <span
                            class={`proof-pill small ${selectedProofStatus}`}
                            title={`Proof ${selectedProofStatus}. Encourage seller to attach evidence when pending/unknown.`}
                          >
                            Proof {selectedProofStatus}
                          </span>
                          {#if selectedProofStatus !== 'fulfilled'}
                            <button class="request-link" type="button" on:click={() => requestProofFor(tx)}>
                              Request proof
                            </button>
                          {/if}
                        {/if}
                        {#if tx.risk_snapshot_hash || riskFor(tx)}
                          <p class="audit-delta">
                            Current risk: {(riskFor(tx)?.score || 0).toFixed(2)}
                            {#if riskCache[tx.listing_hash || tx.id]?.score !== undefined}
                              · Snapshot: {(riskCache[tx.listing_hash || tx.id]?.score || 0).toFixed(2)}
                              ({((riskFor(tx)?.score || 0) - (riskCache[tx.listing_hash || tx.id]?.score || 0)).toFixed(2)})
                            {/if}
                          </p>
                        {/if}
                      </div>
                      <span class="audit-date">{formatDateTime(audit.created_at)}</span>
                    </div>
                    <div class="audit-grid">
                      <div>
                        <p class="audit-label">Proof state at override</p>
                        {#if Object.keys(audit.proof_states || {}).length === 0}
                          <p class="audit-muted">No proof records.</p>
                        {:else}
                          <ul class="audit-listing-list">
                            {#each Object.entries(audit.proof_states) as [listing, state]}
                              <li>
                                <code>{listing.slice(0, 10)}...</code>
                                <span
                                  class={`proof-pill small ${state || 'unknown'}`}
                                  title={`Proof ${state || 'unknown'}`}
                                >
                                  {state || 'unknown'}
                                </span>
                              </li>
                            {/each}
                          </ul>
                        {/if}
                      </div>
                      <div>
                        <p class="audit-label">Risk flags</p>
                        {#if Object.keys(audit.risk_flags || {}).length === 0}
                          <p class="audit-muted">No risk notes.</p>
                        {:else}
                          <ul class="audit-listing-list">
                            {#each Object.entries(audit.risk_flags) as [listing, flag]}
                              <li>
                                <code>{listing.slice(0, 10)}...</code>
                                <span class="audit-flag">{flag || 'Potential anomaly detected'}</span>
                              </li>
                            {/each}
                          </ul>
                        {/if}
                      </div>
                      {#if audit.transaction_ids?.length}
                        <div>
                          <p class="audit-label">Transactions linked</p>
                          <ul class="audit-tx-list">
                            {#each audit.transaction_ids as id}
                              <li><code>{id.substring(0, 12)}...</code></li>
                            {/each}
                          </ul>
                        </div>
                      {/if}
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
          </div>

          <!-- Actions -->
          <div class="modal-section">
            {#if actionError}
              <div class="alert alert-error">
                <span class="alert-icon">⚠️</span>
                {actionError}
              </div>
            {/if}

            {#if actionSuccess}
              <div class="alert alert-success">
                <span class="alert-icon">✓</span>
                {actionSuccess}
              </div>
            {/if}

            <div class="action-buttons">
              {#if tx.can_confirm_delivery}
                <button
                  class="btn btn-success"
                  on:click={confirmDelivery}
                  disabled={actionInProgress || !$isConnected}
                  title={$isConnected ? '' : 'Connect to Holochain to confirm delivery'}
                >
                  ✓ Confirm Delivery
                </button>
              {/if}

              {#if tx.can_leave_review}
                <button
                  class="btn btn-primary"
                  on:click={() => goToReview(tx)}
                  disabled={!$isConnected}
                  title={$isConnected ? '' : 'Connect to Holochain to leave a review'}
                >
                  ⭐ Leave Review
                </button>
              {/if}

              {#if tx.can_file_dispute}
                <button
                  class="btn btn-danger"
                  on:click={() => goToDispute(tx)}
                  disabled={!$isConnected}
                  title={$isConnected ? '' : 'Connect to Holochain to file a dispute'}
                >
                  ⚠️ File Dispute
                </button>
              {/if}

              {#if tx.can_mark_shipped}
                <button
                  class="btn btn-primary"
                  on:click={markAsShipped}
                  disabled={actionInProgress || !$isConnected}
                  title={$isConnected ? '' : 'Connect to Holochain to update shipping'}
                >
                  📦 Mark as Shipped
                </button>
              {/if}
            </div>
          </div>

          <!-- Risk Insights -->
          <div class="modal-section">
            <h3>Why this risk?</h3>
            <RiskInsightDrawer risk={riskFor(tx)} proof={selectedProofStatus} open={false} />
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
.transactions-page {
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
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 3rem;
    background: white;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    color: #742a2a;
  }

  .error-icon {
    font-size: 3rem;
  }

  /* Page Header */
  .page-header {
    margin-bottom: 2rem;
  }

  .page-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .page-header p {
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

  .header-actions {
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }

  .btn-tertiary {
    background: #edf2f7;
    color: #1a202c;
    border: 1px solid #cbd5e0;
  }

  .btn-tertiary:hover:not(:disabled) {
    background: #e2e8f0;
  }

  .proof-pill {
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-weight: 700;
    text-transform: capitalize;
    font-size: 0.75rem;
    border: 1px solid transparent;
    margin-left: 0.35rem;
  }

  .proof-pill.pending,
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

  .risk-pill {
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-weight: 700;
    text-transform: capitalize;
    font-size: 0.75rem;
    border: 1px solid #fed7aa;
    background: #fff7ed;
    color: #c2410c;
  }

  .risk-change {
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.7rem;
    border: 1px solid #a78bfa;
    background: #ede9fe;
    color: #5b21b6;
  }

  .request-link {
    background: transparent;
    border: none;
    color: #2563eb;
    font-weight: 700;
    cursor: pointer;
    padding: 0;
  }

  .request-link:hover {
    text-decoration: underline;
  }

  /* Filters Bar */
  .filters-bar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .trust-toggle {
    gap: 0.35rem;
  }

  .trust-toggle input {
    accent-color: #c05621;
  }

  .trust-count {
    font-size: 0.8rem;
    color: #a0aec0;
  }

  .filter-group label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #4a5568;
  }

  .filter-group select {
    padding: 0.5rem 1rem;
    border: 1px solid #cbd5e0;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }

  .results-count {
    margin-left: auto;
    font-size: 0.875rem;
    color: #718096;
  }

  .filter-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #fffaf0;
    border: 1px solid #f6ad55;
    color: #9a3412;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    gap: 0.5rem;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    background: white;
    border-radius: 0.5rem;
    padding: 3rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .empty-state span {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .empty-state p {
    color: #718096;
    font-size: 1.125rem;
    margin-bottom: 2rem;
  }

  /* Transactions List */
  .transactions-list {
    display: grid;
    gap: 1rem;
  }

  .transaction-card {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1.5rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .transaction-card:hover {
    border-color: #4299e1;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .transaction-card.selected {
    border-color: #4299e1;
    background: #ebf8ff;
  }

  .transaction-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .card-actions {
    display: flex;
    gap: 0.35rem;
    align-items: center;
  }

  .transaction-type {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .preview-pill {
    background: #fffaf0;
    color: #dd6b20;
    border: 1px solid #f6ad55;
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
  }

  .type-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .type-purchase {
    background: #e6f2ff;
    color: #2c5282;
  }

  .type-sale {
    background: #f0fff4;
    color: #276749;
  }

  .override-pill {
    background: #fff7ed;
    color: #9a3412;
    border: 1px solid #fdba74;
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
  }

  .status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .status-warning {
    background: #feebc8;
    color: #7c2d12;
  }

  .status-info {
    background: #bee3f8;
    color: #2c5282;
  }

  .status-success {
    background: #c6f6d5;
    color: #22543d;
  }

  .status-error {
    background: #fed7d7;
    color: #742a2a;
  }

  .transaction-content {
    display: grid;
    grid-template-columns: 80px 1fr;
    gap: 1rem;
  }

  .transaction-thumbnail {
    width: 80px;
    height: 80px;
    border-radius: 0.375rem;
    overflow: hidden;
    background: #f7fafc;
  }

  .transaction-thumbnail img {
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
    color: #cbd5e0;
    font-size: 2rem;
  }

  .transaction-info h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.5rem;
  }

  .transaction-date {
    font-size: 0.875rem;
    color: #a0aec0;
    margin-bottom: 0.5rem;
  }

  .transaction-party {
    font-size: 0.875rem;
    color: #4a5568;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .transaction-price {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38a169;
  }

  .inline-insight {
    margin-top: 0.4rem;
  }

  .override-summary {
    margin-top: 0.35rem;
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
  }

  .override-text {
    margin: 0;
    color: #744210;
    font-size: 0.9rem;
    flex: 1;
  }

  /* Transaction Modal */
  .transaction-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 1rem;
  }

  .modal-content {
    background: white;
    border-radius: 0.5rem;
    padding: 2rem;
    max-width: 700px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
  }

  .modal-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 40px;
    height: 40px;
    border: none;
    background: #e2e8f0;
    border-radius: 50%;
    font-size: 1.5rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .modal-close:hover {
    background: #cbd5e0;
  }

  .modal-content h2 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 1.5rem;
  }

  .modal-section {
    margin-bottom: 2rem;
  }

  .modal-section h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 1rem;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .detail-image {
    width: 100%;
    max-height: 300px;
    object-fit: cover;
    border-radius: 0.375rem;
    margin-bottom: 1.5rem;
  }

  .transaction-details {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: #f7fafc;
    border-radius: 0.375rem;
  }

  .detail-label {
    font-size: 0.875rem;
    color: #718096;
  }

  .detail-value {
    font-size: 0.875rem;
    font-weight: 600;
    color: #2d3748;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  code.detail-value {
    font-family: monospace;
    font-size: 0.75rem;
  }

  /* Timeline */
  .timeline {
    position: relative;
    padding-left: 2.5rem;
  }

  .timeline::before {
    content: '';
    position: absolute;
    left: 0.75rem;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #e2e8f0;
  }

  .timeline-item {
    position: relative;
    padding-bottom: 1.5rem;
  }

  .timeline-item:last-child {
    padding-bottom: 0;
  }

  .timeline-icon {
    position: absolute;
    left: -2rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.875rem;
    color: #718096;
  }

  .timeline-item.completed .timeline-icon {
    background: #38a169;
    color: white;
  }

  .timeline-title {
    font-weight: 600;
    color: #2d3748;
    margin-bottom: 0.25rem;
  }

  .timeline-date {
    font-size: 0.875rem;
    color: #718096;
  }

  .tracking-number {
    font-size: 0.875rem;
    color: #4a5568;
    font-family: monospace;
    margin-top: 0.25rem;
  }

  /* Trust Audit */
  .audit-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .audit-card {
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1rem;
    background: #f8fafc;
  }

  .audit-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
  }

  .audit-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #4a5568;
    margin: 0 0 0.35rem;
  }

  .audit-note {
    margin: 0;
    font-weight: 700;
    color: #1a202c;
  }

  .audit-delta {
    margin: 0.25rem 0 0;
    color: #4a5568;
    font-size: 0.9rem;
  }

  .audit-date {
    font-size: 0.85rem;
    color: #718096;
    white-space: nowrap;
  }

  .audit-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 0.75rem;
  }

  .audit-listing-list,
  .audit-tx-list {
    margin: 0;
    padding-left: 1rem;
    color: #2d3748;
  }

  .audit-listing-list li,
  .audit-tx-list li {
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .audit-flag {
    background: #fff7ed;
    color: #9a3412;
    border: 1px solid #fed7aa;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
  }

  .audit-muted {
    color: #a0aec0;
    margin: 0;
  }

  .audit-empty {
    color: #718096;
    background: #f8fafc;
    border: 1px dashed #cbd5e0;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 0;
  }

  .snapshot-pill {
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.75rem;
    border: 1px solid #cbd5e0;
    background: #edf2f7;
    color: #2d3748;
  }

  .qr-box {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    border: 1px dashed #cbd5e0;
    border-radius: 0.5rem;
    padding: 0.25rem;
    margin-left: 0.5rem;
    font-size: 0.65rem;
    color: #2d3748;
    background: #f8fafc;
  }

  .qr-text {
    font-family: monospace;
    text-align: center;
  }

  .qr-hint {
    font-size: 0.65rem;
    color: #718096;
  }

  .qr-image {
    width: 100px;
    height: 100px;
    margin-left: 0.5rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    background: white;
  }

  .qr-fallback {
    margin-left: 0.5rem;
    color: #718096;
    font-size: 0.8rem;
  }

  @media print {
    /* Hide non-audit UI when printing */
    .transactions-page > .container > *:not(.transaction-modal) {
      display: none !important;
    }

    .transaction-modal {
      position: static;
      background: none;
      padding: 0;
      box-shadow: none;
    }

    .modal-content {
      max-width: none;
      width: 100%;
      max-height: none;
      overflow: visible;
      box-shadow: none;
      padding: 0;
    }

    .modal-close,
    .header-actions,
    .filters-bar,
    .transactions-list,
    .transaction-content > :not(.transaction-info),
    .transaction-details,
    .timeline,
    .action-buttons,
    .inline-insight,
    .card-actions {
      display: none !important;
    }

    .audit-print {
      margin: 0;
      padding: 0;
    }

    .audit-card,
    .audit-list,
    .audit-grid,
    .audit-header {
      page-break-inside: avoid;
    }

    .audit-delta {
      font-weight: 700;
      color: #1a202c;
    }

    .proof-pill,
    .snapshot-pill,
    .risk-change {
      color: #1a202c !important;
      border-color: #1a202c !important;
    }
  }

  .proof-pill.small {
    margin-left: 0;
    padding: 0.15rem 0.45rem;
    font-size: 0.7rem;
  }

  /* Action Buttons */
  .action-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
  }

  /* Buttons */
  .btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 0.375rem;
    font-size: 0.875rem;
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

  .btn-primary:hover:not(:disabled) {
    background: #3182ce;
  }

  .btn-secondary {
    background: #e2e8f0;
    color: #2d3748;
  }

  .btn-secondary:hover:not(:disabled) {
    background: #cbd5e0;
  }

  .btn-success {
    background: #38a169;
    color: white;
  }

  .btn-success:hover:not(:disabled) {
    background: #2f855a;
  }

  .btn-danger {
    background: #e53e3e;
    color: white;
  }

  .btn-danger:hover:not(:disabled) {
    background: #c53030;
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

  .alert-success {
    background: #c6f6d5;
    border: 1px solid #68d391;
    color: #22543d;
  }
</style>
