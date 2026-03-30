<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    enrichedItems,
    purchasingPowerIndex,
    basketHistory,
    hasBasket,
    purchasingPower,
    refreshConsensus,
    submitPriceReport,
    oracleLoading,
    volatilityData,
    basketIndexData,
    exportBasket,
    importBasket,
    CANONICAL_ITEMS,
    RESILIENCE_BASKET_NAME,
    type EnrichedBasketItem,
  } from '$lib/value-basket';
  import { getBalance, type BalanceInfo, type ExchangeRecord } from '$lib/resilience-client';
  import { conductorStatus$ } from '$lib/conductor';
  import { generateTaxExport, exportToCsv, downloadExport } from '$lib/tax-export';
  import { getCommunityConfig } from '$lib/community';
  import { toasts } from '$lib/toast';

  // ============================================================================
  // State
  // ============================================================================

  let balance: BalanceInfo | null = null;

  // Price reporting form
  let reportItem = CANONICAL_ITEMS[0].key;
  let reportPrice = CANONICAL_ITEMS[0].default_price;
  let reportEvidence = '';
  let reportStatus: 'idle' | 'submitting' | 'success' | 'error' | 'cooldown' = 'idle';
  let reportCooldownSecs = 0;
  let cooldownInterval: ReturnType<typeof setInterval> | null = null;

  // Import/export
  let showImportExport = false;
  let importJson = '';
  let importError = '';

  // View mode
  let showConsensusOnly = false;

  // Tax export
  let showTaxExport = false;
  let taxYear = new Date().getFullYear().toString();

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    try {
      balance = await getBalance('self.did');
      // Fetch DHT consensus on load
      await refreshConsensus();
    } catch (e) {
      console.warn('[ValueAnchor] Failed to load data:', e);
    }
  });

  async function handleReport() {
    if (reportPrice <= 0 || !reportItem) return;
    if (reportStatus === 'cooldown') return;
    reportStatus = 'submitting';
    const ok = await submitPriceReport(reportItem, reportPrice, reportEvidence);
    if (ok) {
      reportStatus = 'cooldown';
      reportEvidence = '';
      toasts.success('Price reported');
      reportCooldownSecs = 30;
      cooldownInterval = setInterval(() => {
        reportCooldownSecs -= 1;
        if (reportCooldownSecs <= 0) {
          reportStatus = 'idle';
          if (cooldownInterval) clearInterval(cooldownInterval);
          cooldownInterval = null;
        }
      }, 1000);
    } else {
      reportStatus = 'error';
      toasts.error('Failed to report price');
      setTimeout(() => { reportStatus = 'idle'; }, 2000);
    }
  }

  function handleSelectItem(key: string) {
    reportItem = key;
    const canonical = CANONICAL_ITEMS.find((c) => c.key === key);
    if (canonical) reportPrice = canonical.default_price;
  }

  async function handleRefresh() {
    await refreshConsensus();
  }

  function handleImport() {
    importError = '';
    if (importBasket(importJson)) {
      importJson = '';
      showImportExport = false;
      toasts.success('Basket updated');
    } else {
      importError = 'Invalid JSON format';
      toasts.error('Invalid basket JSON format');
    }
  }

  function handleExport() {
    importJson = exportBasket();
    showImportExport = true;
  }

  function handleTaxExport() {
    const config = getCommunityConfig();
    // In demo mode, generate sample data
    const sampleExchanges: ExchangeRecord[] = [
      { id: 'ex-001', provider_did: 'self.did', receiver_did: 'thandi.did', hours: 2.0, service_description: 'Plumbing repair', service_category: 'HomeServices', status: 'Confirmed', timestamp: Date.now() - 86400000 * 30 },
      { id: 'ex-002', provider_did: 'sipho.did', receiver_did: 'self.did', hours: 1.5, service_description: 'Gardening lessons', service_category: 'Education', status: 'Confirmed', timestamp: Date.now() - 86400000 * 20 },
      { id: 'ex-003', provider_did: 'self.did', receiver_did: 'fatima.did', hours: 3.0, service_description: 'Child minding', service_category: 'CareWork', status: 'Confirmed', timestamp: Date.now() - 86400000 * 10 },
      { id: 'ex-004', provider_did: 'james.did', receiver_did: 'self.did', hours: 2.0, service_description: 'Vehicle maintenance', service_category: 'Transportation', status: 'Confirmed', timestamp: Date.now() - 86400000 * 5 },
    ];

    const summary = generateTaxExport(sampleExchanges, 'self.did', config.dao_did, taxYear);
    const csv = exportToCsv(summary);
    const filename = `mycelix-tax-${config.dao_did}-${taxYear}.csv`;
    downloadExport(csv, filename, 'text/csv');
  }

  $: powerItems = balance ? purchasingPower(balance.balance) : [];

  $: consensusCount = $enrichedItems.filter((i) => i.source === 'consensus').length;
  $: personalCount = $enrichedItems.filter((i) => i.source === 'personal').length;

  $: displayItems = showConsensusOnly
    ? $enrichedItems.filter((i) => i.source === 'consensus')
    : $enrichedItems;

  $: volatilityColor = $volatilityData
    ? $volatilityData.recommended_tier === 'Emergency' ? 'text-red-400'
    : $volatilityData.recommended_tier === 'High' ? 'text-orange-400'
    : $volatilityData.recommended_tier === 'Elevated' ? 'text-yellow-400'
    : 'text-green-400'
    : 'text-gray-400';

  // Sparkline rendering
  function sparkline(history: { timestamp: number; index: number }[]): string {
    if (history.length < 2) return '';
    const blocks = [' ', '\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
    const values = history.slice(-30).map((h) => h.index);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    return values.map((v) => blocks[Math.round(((v - min) / range) * 8)]).join('');
  }
</script>

<svelte:head>
  <title>Value Anchor | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl" aria-hidden="true">&#x2696;</span>
        <div>
          <h1 class="text-lg font-bold">Value Anchor</h1>
          <p class="text-xs text-gray-400">{RESILIENCE_BASKET_NAME}</p>
        </div>
      </div>
      <div class="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-4">
        <!-- Connection status -->
        <div class="text-xs">
          {#if $conductorStatus$ === 'connected'}
            <span class="text-green-400">DHT Live</span>
          {:else}
            <span class="text-yellow-400">Demo Mode</span>
          {/if}
        </div>
        <!-- Volatility tier -->
        {#if $volatilityData}
          <div class="text-right">
            <p class="text-xs text-gray-400">Volatility</p>
            <p class="text-sm font-bold {volatilityColor}">
              {$volatilityData.recommended_tier}
              ({($volatilityData.weekly_change * 100).toFixed(1)}%)
            </p>
          </div>
        {/if}
        <!-- Basket index -->
        {#if $hasBasket}
          <div class="text-right">
            <p class="text-xs text-gray-400">Basket Index</p>
            <p class="text-lg font-bold text-green-400">{$purchasingPowerIndex.toFixed(3)} TEND</p>
          </div>
        {/if}
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Purchasing Power Summary -->
    {#if balance && $hasBasket}
      <div class="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h2 class="text-gray-400 text-xs uppercase mb-3">Your {balance.balance} TEND buys:</h2>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
          {#each powerItems.slice(0, 10) as item}
            <div class="bg-gray-700/50 rounded-lg p-3 text-center">
              <p class="text-xl font-bold text-green-400">{item.quantity}</p>
              <p class="text-xs text-gray-400">{item.unit} {item.name}</p>
            </div>
          {/each}
        </div>
        {#if $basketHistory.length >= 2}
          <div class="mt-4">
            <p class="text-xs text-gray-400 mb-1">Basket index trend (last 30 updates)</p>
            <p class="font-mono text-lg text-green-400 tracking-wide">{sparkline($basketHistory)}</p>
          </div>
        {/if}
      </div>
    {/if}

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Price Table (Consensus + Personal) -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 lg:col-span-2">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <div class="flex items-center gap-4">
            <h2 class="text-lg font-semibold">Community Prices</h2>
            <div class="flex gap-2 text-xs">
              <span class="text-green-400">{consensusCount} consensus</span>
              <span class="text-gray-500">|</span>
              <span class="text-yellow-400">{personalCount} personal</span>
            </div>
          </div>
          <div class="flex gap-2">
            <button on:click={handleRefresh} disabled={$oracleLoading}
              class="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded transition-colors">
              {$oracleLoading ? 'Refreshing...' : 'Refresh DHT'}
            </button>
            <button on:click={() => showConsensusOnly = !showConsensusOnly}
              class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors">
              {showConsensusOnly ? 'Show All' : 'Consensus Only'}
            </button>
            <button on:click={handleExport}
              class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors">
              Export
            </button>
            <button on:click={() => { showImportExport = !showImportExport; importJson = ''; importError = ''; }}
              class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors">
              Import
            </button>
          </div>
        </div>

        {#if showImportExport}
          <div class="p-4 border-b border-gray-700 bg-gray-700/30">
            <textarea bind:value={importJson} rows="4" placeholder="Paste basket JSON here..."
              class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"></textarea>
            {#if importError}
              <p class="text-red-400 text-xs mt-1">{importError}</p>
            {/if}
            <button on:click={handleImport} disabled={!importJson.trim()}
              class="mt-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded px-3 py-1 text-sm transition-colors">
              Import Basket
            </button>
          </div>
        {/if}

        <!-- Price list -->
        <div class="p-4 space-y-2">
          {#each displayItems as item (item.key)}
            <div class="p-3 bg-gray-700/50 rounded-lg flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2">
              <div class="flex items-center gap-3">
                <!-- Source indicator -->
                <div class="w-2 h-2 rounded-full {item.source === 'consensus' ? 'bg-green-400' : 'bg-yellow-400'}"
                  title="{item.source === 'consensus' ? 'DHT Consensus' : 'Personal Estimate'}"
                  aria-hidden="true"></div>
                <div>
                  <p class="font-medium text-sm">{item.name}</p>
                  <p class="text-xs text-gray-400">{item.unit}</p>
                </div>
              </div>
              <div class="flex items-center gap-4">
                {#if item.source === 'consensus'}
                  <div class="text-right">
                    <span class="text-sm font-bold text-green-400">{item.consensus_price?.toFixed(2)} TEND</span>
                    <p class="text-xs text-gray-500">
                      {item.reporter_count} reporters | SD {item.std_dev.toFixed(3)} |
                      <span class="{item.signal_integrity > 0.8 ? 'text-green-400' : item.signal_integrity > 0.5 ? 'text-yellow-400' : 'text-red-400'}">
                        SI {(item.signal_integrity * 100).toFixed(0)}%
                      </span>
                    </p>
                  </div>
                {:else}
                  <span class="text-sm font-bold text-yellow-400">{item.price_tend.toFixed(2)} TEND</span>
                {/if}
                <button on:click={() => handleSelectItem(item.key)}
                  class="text-xs text-blue-400 hover:text-blue-300"
                  aria-label="Report price for {item.name}">report</button>
              </div>
            </div>
          {:else}
            <div class="text-center py-8">
              <p class="text-gray-500 mb-2">No price data yet</p>
              <p class="text-xs text-gray-500">Be the first to report a price below</p>
            </div>
          {/each}
        </div>
      </div>

      <!-- Right Column: Report + Basket Index -->
      <div class="space-y-6">
        <!-- Price Reporter -->
        <div class="bg-gray-800 rounded-lg border border-green-700/50 p-4">
          <h3 class="text-sm font-semibold text-green-300 mb-3">Signal a Local Price</h3>
          <p class="text-xs text-gray-400 mb-3">
            Report what you paid today. Your signal joins the community consensus.
          </p>
          <form on:submit|preventDefault={handleReport} class="space-y-3">
            <div>
              <label for="report-item" class="text-xs text-gray-400">Item</label>
              <select id="report-item" bind:value={reportItem}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500">
                {#each CANONICAL_ITEMS as item}
                  <option value={item.key}>{item.name}</option>
                {/each}
              </select>
            </div>
            <div>
              <label for="report-price" class="text-xs text-gray-400">Price (TEND)</label>
              <input id="report-price" type="number" bind:value={reportPrice}
                min="0.01" step="0.01"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
              <p class="text-xs text-gray-500 mt-1">1 TEND = 1 hour of labor</p>
            </div>
            <div>
              <label for="report-evidence" class="text-xs text-gray-400">Evidence (optional)</label>
              <input id="report-evidence" type="text" bind:value={reportEvidence}
                placeholder="e.g. Local shop, 2026-03-14"
                maxlength="512"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500" />
            </div>
            <button type="submit"
              disabled={reportPrice <= 0 || reportStatus === 'submitting'}
              class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-3 py-2 text-sm font-medium transition-colors">
              {#if reportStatus === 'submitting'}
                Broadcasting...
              {:else if reportStatus === 'success'}
                Signal Sent
              {:else if reportStatus === 'error'}
                Failed (Demo Mode?)
              {:else}
                Broadcast Price Signal
              {/if}
            </button>
          </form>
        </div>

        <!-- Basket Index Detail -->
        {#if $basketIndexData}
          <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 class="text-sm font-semibold text-gray-300 mb-3">Basket Breakdown</h3>
            <div class="space-y-2">
              {#each $basketIndexData.item_prices as ip}
                <div class="flex justify-between text-xs">
                  <span class="text-gray-400">{ip.item}</span>
                  <div class="text-right">
                    <span class="text-white">{ip.price.toFixed(2)}</span>
                    <span class="text-gray-500"> x {(ip.weight * 100).toFixed(0)}%</span>
                    <span class="text-green-400 ml-1">= {ip.weighted_price.toFixed(3)}</span>
                  </div>
                </div>
              {/each}
              <div class="border-t border-gray-700 pt-2 flex justify-between text-sm font-bold">
                <span>Total Index</span>
                <span class="text-green-400">{$basketIndexData.index.toFixed(3)} TEND</span>
              </div>
            </div>
          </div>
        {/if}

        <!-- Volatility Monitor -->
        {#if $volatilityData}
          <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 class="text-sm font-semibold text-gray-300 mb-2">Economic Pulse</h3>
            <div class="space-y-2 text-xs">
              <div class="flex justify-between">
                <span class="text-gray-400">Weekly Change</span>
                <span class="{volatilityColor} font-bold">
                  {($volatilityData.weekly_change * 100).toFixed(1)}%
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Tier</span>
                <span class="{volatilityColor}">{$volatilityData.recommended_tier}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">TEND Limit Escalated</span>
                <span class="{$volatilityData.escalated ? 'text-red-400' : 'text-gray-500'}">
                  {$volatilityData.escalated ? 'Yes' : 'No'}
                </span>
              </div>
              {#if $volatilityData.recommended_tier !== 'Normal'}
                <p class="text-yellow-400 mt-2">
                  TEND credit limits adjusted to {$volatilityData.recommended_tier} tier.
                </p>
              {/if}
            </div>
          </div>
        {/if}

        <!-- Tax Export -->
        <div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 class="text-sm font-semibold text-gray-300 mb-2">Tax Record Export</h3>
          <p class="text-xs text-gray-400 mb-3">
            Download TEND exchange history valued in {getCommunityConfig().currency_code}
            for {getCommunityConfig().tax_authority} ({getCommunityConfig().tax_form_name}) reporting.
          </p>
          <div class="flex gap-2 mb-3">
            <div class="flex-1">
              <label for="tax-year" class="text-xs text-gray-400">Tax Year</label>
              <input id="tax-year" type="number" bind:value={taxYear}
                min="2020" max="2030"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500" />
            </div>
          </div>
          <button on:click={handleTaxExport}
            class="w-full bg-gray-600 hover:bg-gray-500 rounded px-3 py-2 text-sm font-medium transition-colors">
            Download CSV
          </button>
          <p class="text-xs text-gray-500 mt-2">
            Valued at {getCommunityConfig().currency_symbol}{getCommunityConfig().labor_hour_value}/hour
            ({getCommunityConfig().labor_hour_source})
          </p>
        </div>

        <!-- How It Works -->
        <div class="bg-gray-800/50 rounded-lg border border-gray-700/50 p-4">
          <h3 class="text-sm font-semibold text-gray-400 mb-2">How Consensus Works</h3>
          <ul class="text-xs text-gray-500 space-y-1">
            <li>Community members report local prices</li>
            <li>Top/bottom 10% are trimmed (anti-manipulation)</li>
            <li>Accuracy-weighted median = consensus price</li>
            <li>Reporters weighted by historical accuracy (EMA)</li>
            <li>SI% = Signal Integrity (avg reporter accuracy)</li>
            <li>Min 2 reporters required per item</li>
            <li>7-day rolling window</li>
            <li>20%+ weekly swing = TEND limit escalation</li>
          </ul>
        </div>
      </div>
    </div>

    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Value Anchor &middot; Community-driven price consensus on Holochain DHT</p>
    </footer>
  </main>
</div>
