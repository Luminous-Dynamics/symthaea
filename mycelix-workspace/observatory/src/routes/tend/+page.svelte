<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import {
    getBalance,
    getOracleState,
    getDaoListings,
    getDaoRequests,
    recordExchange,
    type BalanceInfo,
    type OracleState,
    type ServiceListing,
    type ServiceRequest,
    type ExchangeRecord,
  } from '$lib/resilience-client';
  import { hasBasket, purchasingPowerIndex, enrichedItems } from '$lib/value-basket';
  import { purchasingPower } from '$lib/value-basket';
  import { toasts } from '$lib/toast';
  import { exportTendExchangesCsv } from '$lib/data-export';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';

  function exportListings() {
    const records: ExchangeRecord[] = $listings.map(l => ({
      id: l.id, provider_did: l.provider_did, receiver_did: '', hours: l.hours_estimate,
      service_description: l.description, service_category: l.category, status: 'Confirmed' as const, timestamp: l.created_at
    }));
    exportTendExchangesCsv(records);
  }

  // ============================================================================
  // Stores
  // ============================================================================

  const balance = writable<BalanceInfo | null>(null);
  const oracle = writable<OracleState>({ vitality: 72, tier: 'Normal', updated_at: Date.now() });
  const listings = writable<ServiceListing[]>([]);
  const requests = writable<ServiceRequest[]>([]);
  const lastExchange = writable<ExchangeRecord | null>(null);

  // Form state
  let receiverDid = '';
  let hours = 1;
  let serviceDesc = '';
  let serviceCategory = 'General';
  let submitting = false;
  let loading = true;
  let tab: 'listings' | 'requests' = 'listings';

  // Search & filter
  let searchQuery = '';
  let filterCategory = '';

  const categories = ['General', 'Maintenance', 'Education', 'Childcare', 'Transport', 'Food', 'Care', 'Construction', 'Tech'];

  function matchesSearch(...fields: string[]): boolean {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return fields.some(f => f && f.toLowerCase().includes(q));
  }

  function matchesCategory(category: string): boolean {
    return !filterCategory || category === filterCategory;
  }

  $: filteredListings = $listings.filter(l => matchesSearch(l.title, l.description) && matchesCategory(l.category));
  $: filteredRequests = $requests.filter(r => matchesSearch(r.title, r.description) && matchesCategory(r.category));

  // ============================================================================
  // Freshness — 2min polling
  // ============================================================================

  async function fetchData() {
    const [bal, ora, lst, req] = await Promise.all([
      getBalance('self.did'),
      getOracleState(),
      getDaoListings(),
      getDaoRequests(),
    ]);
    balance.set(bal);
    oracle.set(ora);
    listings.set(lst);
    requests.set(req);
  }

  const freshness = createFreshness(fetchData, 120_000);
  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } = freshness;

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    await refresh();
    loading = false;
    startPolling();
  });

  onDestroy(() => stopPolling());

  async function handleExchange() {
    if (!receiverDid || hours <= 0 || !serviceDesc) return;
    submitting = true;
    try {
      const ex = await recordExchange(receiverDid, hours, serviceDesc, serviceCategory);
      lastExchange.set(ex);
      // Refresh balance
      const bal = await getBalance('self.did');
      balance.set(bal);
      receiverDid = '';
      hours = 1;
      serviceDesc = '';
      toasts.success('Exchange recorded successfully');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to record exchange');
    } finally {
      submitting = false;
    }
  }

  function tierColor(tier: string): string {
    switch (tier) {
      case 'Normal': return 'text-green-400';
      case 'Elevated': return 'text-yellow-400';
      case 'High': return 'text-orange-400';
      case 'Emergency': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }

  function tierBg(tier: string): string {
    switch (tier) {
      case 'Normal': return 'bg-green-900/30 border-green-700';
      case 'Elevated': return 'bg-yellow-900/30 border-yellow-700';
      case 'High': return 'bg-orange-900/30 border-orange-700';
      case 'Emergency': return 'bg-red-900/30 border-red-700';
      default: return 'bg-gray-900/30 border-gray-700';
    }
  }
</script>

<svelte:head>
  <title>TEND | Mycelix Observatory</title>
</svelte:head>

{#if loading}
  <div class="text-white p-8 text-center text-gray-400">Loading TEND data...</div>
{:else}
<div class="text-white">
  <!-- Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl" aria-hidden="true">&#x1F91D;</span>
        <div>
          <h1 class="text-lg font-bold">TEND — Time Exchange</h1>
          <p class="text-xs text-gray-400">All hours are equal. 1 TEND = 1 hour of labor.</p>
        </div>
      </div>
      {#if $oracle}
        <div class="flex items-center gap-3">
          <div class={`px-3 py-1 rounded border text-sm font-medium ${tierBg($oracle.tier)}`}>
            <span class={tierColor($oracle.tier)}>Oracle: {$oracle.tier}</span>
          </div>
          <div class="text-right">
            <p class="text-xs text-gray-400">Limit</p>
            <p class="text-sm font-bold">
              {#if $oracle.tier === 'Normal'}&plusmn;40
              {:else if $oracle.tier === 'Elevated'}&plusmn;60
              {:else if $oracle.tier === 'High'}&plusmn;80
              {:else}&plusmn;120
              {/if} TEND
            </p>
          </div>
        </div>
      {/if}
    </div>
  </header>

  <main class="container mx-auto p-6">
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />
    <!-- Balance + Record Exchange -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
      <!-- Balance Card -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h2 class="text-gray-400 text-xs uppercase mb-4">Your Balance</h2>
        {#if $balance}
          <p class="text-4xl font-bold" class:text-green-400={$balance.balance > 0} class:text-red-400={$balance.balance < 0}>
            {$balance.balance > 0 ? '+' : ''}{$balance.balance} TEND
          </p>
          <div class="mt-4 grid grid-cols-3 gap-2 text-center">
            <div>
              <p class="text-lg font-semibold text-green-400">{$balance.total_earned}</p>
              <p class="text-xs text-gray-400">Earned</p>
            </div>
            <div>
              <p class="text-lg font-semibold text-orange-400">{$balance.total_spent}</p>
              <p class="text-xs text-gray-400">Spent</p>
            </div>
            <div>
              <p class="text-lg font-semibold">{$balance.exchange_count}</p>
              <p class="text-xs text-gray-400">Exchanges</p>
            </div>
          </div>
          {#if $hasBasket && $balance}
            <div class="mt-4 pt-3 border-t border-gray-700">
              <p class="text-xs text-gray-400 mb-2">Purchasing Power</p>
              <div class="space-y-1">
                {#each purchasingPower($balance.balance).slice(0, 4) as item}
                  <div class="flex justify-between text-xs">
                    <span class="text-gray-300">{item.name}</span>
                    <span class="text-green-400 font-medium">{item.quantity} {item.unit}</span>
                  </div>
                {/each}
              </div>
              <a href="/value-anchor" class="block mt-2 text-xs text-blue-400 hover:text-blue-300">Edit basket</a>
            </div>
          {/if}
        {:else}
          <p class="text-gray-500">Loading...</p>
        {/if}
      </div>

      <!-- Record Exchange Form -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 lg:col-span-2">
        <h2 class="text-gray-400 text-xs uppercase mb-4">Record Exchange</h2>
        {#if $lastExchange}
          <div class="mb-4 p-3 bg-green-900/30 border border-green-700 rounded-lg text-sm">
            Exchange recorded: {$lastExchange.hours}h to {$lastExchange.receiver_did} — {$lastExchange.service_description}
          </div>
        {/if}
        <form on:submit|preventDefault={handleExchange} class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="receiver" class="text-xs text-gray-400">Receiver DID</label>
            <input id="receiver" bind:value={receiverDid} placeholder="member.did" required
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500" />
          </div>
          <div>
            <label for="hours" class="text-xs text-gray-400">Hours</label>
            <input id="hours" type="number" bind:value={hours} min="0.25" step="0.25" max="12" required
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500" />
          </div>
          <div>
            <label for="desc" class="text-xs text-gray-400">Service Description</label>
            <input id="desc" bind:value={serviceDesc} placeholder="What was done?" required
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500" />
          </div>
          <div>
            <label for="cat" class="text-xs text-gray-400">Category</label>
            <select id="cat" bind:value={serviceCategory}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500">
              {#each categories as cat}
                <option value={cat}>{cat}</option>
              {/each}
            </select>
          </div>
          <div class="md:col-span-2">
            <button type="submit" disabled={submitting || !receiverDid || !serviceDesc}
              class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
              {submitting ? 'Recording...' : 'Record Exchange'}
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- Search & Category Filter -->
    <div class="mb-6 space-y-3">
      <input type="text" bind:value={searchQuery} placeholder="Search listings & requests..."
        aria-label="Search TEND listings and requests"
        class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500" />
      <div class="flex flex-wrap gap-2">
        <button on:click={() => filterCategory = ''}
          class="px-3 py-1 rounded-full text-sm transition-colors {filterCategory === '' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}">
          All
        </button>
        {#each categories as cat}
          <button on:click={() => filterCategory = cat}
            class="px-3 py-1 rounded-full text-sm transition-colors {filterCategory === cat ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}">
            {cat}
          </button>
        {/each}
      </div>
    </div>

    <!-- Marketplace Tabs -->
    <div class="bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700 flex items-center gap-4">
        <h2 class="text-lg font-semibold">Service Marketplace</h2>
        <div class="flex gap-1 ml-auto">
          <button on:click={() => tab = 'listings'}
            class="px-3 py-1 rounded text-sm {tab === 'listings' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'} transition-colors">
            Listings ({filteredListings.length})
          </button>
          <button on:click={() => tab = 'requests'}
            class="px-3 py-1 rounded text-sm {tab === 'requests' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'} transition-colors">
            Requests ({filteredRequests.length})
          </button>
        </div>
      </div>
      <div class="p-4 space-y-3">
        {#if tab === 'listings'}
          {#each filteredListings as listing}
            <div class="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <p class="font-medium">{listing.title}</p>
                  <p class="text-xs text-gray-400 mt-1">{listing.description}</p>
                </div>
                <div class="text-right">
                  <span class="text-sm font-bold text-green-400">{listing.hours_estimate}h</span>
                  <p class="text-xs text-gray-400 mt-1">{listing.category}</p>
                </div>
              </div>
              <div class="flex justify-between text-xs text-gray-500 mt-2">
                <span>by {listing.provider_did}</span>
                <span>{new Date(listing.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-8">{searchQuery || filterCategory ? 'No matching listings' : 'No listings yet'}</p>
          {/each}
        {:else}
          {#each filteredRequests as request}
            <div class="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <div class="flex items-center gap-2">
                    <p class="font-medium">{request.title}</p>
                    <span class="text-xs px-2 py-0.5 rounded {request.urgency === 'High' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}">
                      {request.urgency}
                    </span>
                  </div>
                  <p class="text-xs text-gray-400 mt-1">{request.description}</p>
                </div>
                <div class="text-right">
                  <span class="text-sm font-bold text-orange-400">{request.hours_budget}h budget</span>
                  <p class="text-xs text-gray-400 mt-1">{request.category}</p>
                </div>
              </div>
              <div class="flex justify-between text-xs text-gray-500 mt-2">
                <span>by {request.requester_did}</span>
                <span>{new Date(request.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-8">{searchQuery || filterCategory ? 'No matching requests' : 'No requests yet'}</p>
          {/each}
        {/if}
      </div>
    </div>

    <!-- TEND Mechanics -->
    <div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-sm font-semibold mb-3 text-gray-300">Mutual Credit</h3>
        <p class="text-xs text-gray-400">Zero-sum: every credit = equal debit elsewhere. No money creation, no inflation. Community credit always sums to zero.</p>
      </div>
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-sm font-semibold mb-3 text-gray-300">Radical Equality</h3>
        <p class="text-xs text-gray-400">1 TEND = 1 hour, regardless of service. A doctor's hour equals a gardener's hour. Dignity in all labor.</p>
      </div>
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <h3 class="text-sm font-semibold mb-3 text-gray-300">Oracle Tiers</h3>
        <p class="text-xs text-gray-400">Credit limits adjust with community stress. Normal &plusmn;40, Elevated &plusmn;60, High &plusmn;80, Emergency &plusmn;120 TEND.</p>
      </div>
    </div>

    <div class="mt-6 flex justify-end">
      <button on:click={exportListings}
        class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
        Export Listings CSV
      </button>
    </div>

    <footer class="mt-4 text-center text-gray-500 text-sm">
      <p>TEND v1.0 &middot; Commons Charter Article II, Section 2 &middot; All Hours Are Equal</p>
    </footer>
  </main>
</div>
{/if}
