<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import {
    getServiceOffers,
    getServiceRequests,
    createServiceOffer,
    createServiceRequest,
    type AidOffer,
    type AidRequest,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';
  import { exportAidOffersCsv, exportAidRequestsCsv } from '$lib/data-export';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';

  // ============================================================================
  // Stores
  // ============================================================================

  const offers = writable<AidOffer[]>([]);
  const requests = writable<AidRequest[]>([]);

  let tab: 'offers' | 'requests' = 'offers';

  // Offer form
  let offerTitle = '';
  let offerDesc = '';
  let offerCategory = 'Care';
  let offerHours = 2;
  let showOfferForm = false;

  // Request form
  let reqTitle = '';
  let reqDesc = '';
  let reqCategory = 'Care';
  let reqUrgency: AidRequest['urgency'] = 'medium';
  let reqHours = 2;
  let showRequestForm = false;

  let submitting = false;
  let loading = true;

  // Search & filter
  let searchQuery = '';
  let filterCategory = '';

  const categories = ['Care', 'Transport', 'Food', 'Maintenance', 'Education', 'Childcare', 'Tech', 'Admin', 'Other'];
  const urgencyLevels: AidRequest['urgency'][] = ['low', 'medium', 'high', 'critical'];

  function matchesSearch(...fields: string[]): boolean {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return fields.some(f => f && f.toLowerCase().includes(q));
  }

  function matchesCategory(category: string): boolean {
    return !filterCategory || category === filterCategory;
  }

  $: filteredOffers = $offers.filter(o => matchesSearch(o.title, o.description) && matchesCategory(o.category));
  $: filteredRequests = $requests.filter(r => matchesSearch(r.title, r.description) && matchesCategory(r.category));

  // ============================================================================
  // Freshness — 2min polling
  // ============================================================================

  async function fetchData() {
    const [o, r] = await Promise.all([
      getServiceOffers(),
      getServiceRequests(),
    ]);
    offers.set(o);
    requests.set(r);
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

  async function handleCreateOffer() {
    if (!offerTitle || !offerDesc) return;
    submitting = true;
    try {
      const o = await createServiceOffer(offerTitle, offerDesc, offerCategory, offerHours);
      offers.update(list => [o, ...list]);
      offerTitle = '';
      offerDesc = '';
      showOfferForm = false;
      toasts.success('Offer posted');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to post offer');
    } finally {
      submitting = false;
    }
  }

  async function handleCreateRequest() {
    if (!reqTitle || !reqDesc) return;
    submitting = true;
    try {
      const r = await createServiceRequest(reqTitle, reqDesc, reqCategory, reqUrgency, reqHours);
      requests.update(list => [r, ...list]);
      reqTitle = '';
      reqDesc = '';
      showRequestForm = false;
      toasts.success('Request posted');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to post request');
    } finally {
      submitting = false;
    }
  }

  function urgencyColor(u: string): string {
    switch (u) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  }
</script>

<svelte:head>
  <title>Mutual Aid | Mycelix Observatory</title>
</svelte:head>

{#if loading}
  <div class="text-white p-8 text-center text-gray-400">Loading mutual aid data...</div>
{:else}
<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl" aria-hidden="true">&#x1F932;</span>
        <div>
          <h1 class="text-lg font-bold">Mutual Aid Timebank</h1>
          <p class="text-xs text-gray-400">Community care, neighbor to neighbor</p>
        </div>
      </div>
      <div class="flex items-center gap-4">
        <div class="text-right">
          <p class="text-xs text-gray-400">Active Offers</p>
          <p class="text-lg font-bold text-green-400">{$offers.length}</p>
        </div>
        <div class="text-right">
          <p class="text-xs text-gray-400">Open Requests</p>
          <p class="text-lg font-bold text-orange-400">{$requests.filter(r => !r.fulfilled).length}</p>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />
    <!-- Stats -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Active Offers</h3>
        <p class="text-2xl font-bold mt-1 text-green-400">{$offers.length}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Open Requests</h3>
        <p class="text-2xl font-bold mt-1 text-orange-400">{$requests.filter(r => !r.fulfilled).length}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Hours Offered</h3>
        <p class="text-2xl font-bold mt-1 text-blue-400">{$offers.reduce((s, o) => s + o.hours_available, 0)}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Hours Needed</h3>
        <p class="text-2xl font-bold mt-1 text-purple-400">{$requests.filter(r => !r.fulfilled).reduce((s, r) => s + r.hours_needed, 0)}</p>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex gap-3 mb-6">
      <button on:click={() => { showOfferForm = !showOfferForm; showRequestForm = false; }}
        class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors">
        + Offer Help
      </button>
      <button on:click={() => { showRequestForm = !showRequestForm; showOfferForm = false; }}
        class="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm font-medium transition-colors">
        + Request Help
      </button>
    </div>

    <!-- Search & Category Filter -->
    <div class="mb-6 space-y-3">
      <input type="text" bind:value={searchQuery} placeholder="Search offers & requests..."
        aria-label="Search mutual aid offers and requests"
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

    <!-- Offer Form -->
    {#if showOfferForm}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-sm font-semibold text-gray-300 mb-4">Offer Help</h2>
        <form on:submit|preventDefault={handleCreateOffer} class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="otitle" class="text-xs text-gray-400">What can you help with?</label>
            <input id="otitle" bind:value={offerTitle} placeholder="e.g. Plumbing repair" required
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
          </div>
          <div>
            <label for="ocat" class="text-xs text-gray-400">Category</label>
            <select id="ocat" bind:value={offerCategory}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500">
              {#each categories as c}
                <option value={c}>{c}</option>
              {/each}
            </select>
          </div>
          <div class="md:col-span-2">
            <label for="odesc" class="text-xs text-gray-400">Description</label>
            <textarea id="odesc" bind:value={offerDesc} rows="2" placeholder="Describe what you can do..."
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"></textarea>
          </div>
          <div>
            <label for="ohours" class="text-xs text-gray-400">Hours available</label>
            <input id="ohours" type="number" bind:value={offerHours} min="0.5" step="0.5" max="24"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
          </div>
          <div class="flex items-end">
            <button type="submit" disabled={submitting || !offerTitle || !offerDesc}
              class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
              {submitting ? 'Posting...' : 'Post Offer'}
            </button>
          </div>
        </form>
      </div>
    {/if}

    <!-- Request Form -->
    {#if showRequestForm}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-sm font-semibold text-gray-300 mb-4">Request Help</h2>
        <form on:submit|preventDefault={handleCreateRequest} class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="rtitle" class="text-xs text-gray-400">What do you need?</label>
            <input id="rtitle" bind:value={reqTitle} placeholder="e.g. Help installing gas stove" required
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-orange-500" />
          </div>
          <div>
            <label for="rcat" class="text-xs text-gray-400">Category</label>
            <select id="rcat" bind:value={reqCategory}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-orange-500">
              {#each categories as c}
                <option value={c}>{c}</option>
              {/each}
            </select>
          </div>
          <div class="md:col-span-2">
            <label for="rdesc" class="text-xs text-gray-400">Description</label>
            <textarea id="rdesc" bind:value={reqDesc} rows="2" placeholder="Describe what you need..."
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-orange-500"></textarea>
          </div>
          <div>
            <label for="rurgency" class="text-xs text-gray-400">Urgency</label>
            <select id="rurgency" bind:value={reqUrgency}
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-orange-500">
              {#each urgencyLevels as u}
                <option value={u}>{u.charAt(0).toUpperCase() + u.slice(1)}</option>
              {/each}
            </select>
          </div>
          <div>
            <label for="rhours" class="text-xs text-gray-400">Hours needed (estimate)</label>
            <input id="rhours" type="number" bind:value={reqHours} min="0.5" step="0.5" max="24"
              class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-orange-500" />
          </div>
          <div class="md:col-span-2">
            <button type="submit" disabled={submitting || !reqTitle || !reqDesc}
              class="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
              {submitting ? 'Posting...' : 'Post Request'}
            </button>
          </div>
        </form>
      </div>
    {/if}

    <!-- Tabs -->
    <div class="bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700 flex items-center gap-4">
        <div class="flex gap-1">
          <button on:click={() => tab = 'offers'}
            class="px-3 py-1 rounded text-sm {tab === 'offers' ? 'bg-green-600' : 'bg-gray-700 hover:bg-gray-600'} transition-colors">
            Offers ({filteredOffers.length})
          </button>
          <button on:click={() => tab = 'requests'}
            class="px-3 py-1 rounded text-sm {tab === 'requests' ? 'bg-orange-600' : 'bg-gray-700 hover:bg-gray-600'} transition-colors">
            Requests ({filteredRequests.length})
          </button>
        </div>
      </div>
      <div class="p-4 space-y-3">
        {#if tab === 'offers'}
          {#each filteredOffers as offer}
            <div class="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <div class="flex items-center gap-2">
                    <p class="font-medium">{offer.title}</p>
                    <span class="text-xs px-2 py-0.5 rounded bg-gray-600 text-gray-300">{offer.category}</span>
                    {#if offer.recurring}
                      <span class="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/50">Recurring</span>
                    {/if}
                  </div>
                  <p class="text-xs text-gray-400 mt-1">{offer.description}</p>
                </div>
                <div class="text-right">
                  <span class="text-sm font-bold text-green-400">{offer.hours_available}h</span>
                </div>
              </div>
              <div class="flex justify-between text-xs text-gray-500 mt-2">
                <span>by {offer.provider_did}</span>
                <span>{new Date(offer.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-8">{searchQuery || filterCategory ? 'No matching offers' : 'No offers yet. Be the first to help!'}</p>
          {/each}
        {:else}
          {#each filteredRequests as request}
            <div class="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors">
              <div class="flex justify-between items-start">
                <div>
                  <div class="flex items-center gap-2">
                    <p class="font-medium">{request.title}</p>
                    <span class={`text-xs px-2 py-0.5 rounded border ${urgencyColor(request.urgency)}`}>
                      {request.urgency}
                    </span>
                    <span class="text-xs px-2 py-0.5 rounded bg-gray-600 text-gray-300">{request.category}</span>
                  </div>
                  <p class="text-xs text-gray-400 mt-1">{request.description}</p>
                </div>
                <div class="text-right">
                  <span class="text-sm font-bold text-orange-400">{request.hours_needed}h</span>
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

    <div class="mt-6 flex justify-end gap-2">
      <button on:click={() => exportAidOffersCsv($offers)}
        class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
        Export Offers CSV
      </button>
      <button on:click={() => exportAidRequestsCsv($requests)}
        class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
        Export Requests CSV
      </button>
    </div>

    <footer class="mt-4 text-center text-gray-500 text-sm">
      <p>Mutual Aid Timebank &middot; Mycelix Commons</p>
    </footer>
  </main>
</div>
{/if}
