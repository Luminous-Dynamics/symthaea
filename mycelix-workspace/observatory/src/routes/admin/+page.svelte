<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    getBalance,
    getOracleState,
    getDaoListings,
    getDaoRequests,
    getServiceOffers,
    getServiceRequests,
    getChannels,
    getAllPlots,
    getAllWaterSystems,
    getActiveWaterAlerts,
    getLowStockItems,
    type OracleState,
    type LowStockItem,
  } from '$lib/resilience-client';
  import { connectionHealth, connectionQuality, connectionLabel, qualityColor } from '$lib/connection-health';
  import { queueCount, getQueue, removeItem, clearQueue, type QueuedSubmission } from '$lib/offline-queue';
  import { createFreshness } from '$lib/freshness';
  import { getCommunityConfig } from '$lib/community';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';
  import { browser } from '$app/environment';

  // ============================================================================
  // Dashboard data
  // ============================================================================

  let loading = true;

  // Counts
  let tendListings = 0;
  let tendRequests = 0;
  let aidOffers = 0;
  let aidRequests = 0;
  let emergencyChannels = 0;
  let foodPlots = 0;
  let waterSystems = 0;
  let waterAlerts = 0;
  let lowStockItems: LowStockItem[] = [];
  let oracle: OracleState = { vitality: 0, tier: 'Normal', updated_at: 0 };

  // Offline queue
  let queueItems: QueuedSubmission[] = [];

  // Activity log
  type LogEntry = { time: number; domain: string; message: string };
  let activityLog: LogEntry[] = [];

  function log(domain: string, message: string) {
    activityLog = [{ time: Date.now(), domain, message }, ...activityLog].slice(0, 50);
  }

  // ============================================================================
  // Freshness — 30s polling (operator needs current data)
  // ============================================================================

  async function fetchData() {
    const results = await Promise.allSettled([
      getDaoListings(),
      getDaoRequests(),
      getServiceOffers(),
      getServiceRequests(),
      getChannels(),
      getAllPlots(),
      getAllWaterSystems(),
      getActiveWaterAlerts(),
      getLowStockItems(),
      getOracleState(),
    ]);

    const prev = { tendListings, tendRequests, aidOffers, aidRequests, emergencyChannels, foodPlots, waterSystems, waterAlerts };

    if (results[0].status === 'fulfilled') tendListings = results[0].value.length;
    if (results[1].status === 'fulfilled') tendRequests = results[1].value.length;
    if (results[2].status === 'fulfilled') aidOffers = results[2].value.length;
    if (results[3].status === 'fulfilled') aidRequests = results[3].value.length;
    if (results[4].status === 'fulfilled') emergencyChannels = results[4].value.length;
    if (results[5].status === 'fulfilled') foodPlots = results[5].value.length;
    if (results[6].status === 'fulfilled') waterSystems = results[6].value.length;
    if (results[7].status === 'fulfilled') {
      const newAlerts = results[7].value.length;
      if (newAlerts > prev.waterAlerts && prev.waterAlerts > 0) {
        log('water', `${newAlerts - prev.waterAlerts} new water alert(s)`);
      }
      waterAlerts = newAlerts;
    }
    if (results[8].status === 'fulfilled') lowStockItems = results[8].value;
    if (results[9].status === 'fulfilled') oracle = results[9].value;

    queueItems = await getQueue();

    // Log changes
    if (tendListings !== prev.tendListings) log('tend', `TEND listings: ${prev.tendListings} → ${tendListings}`);
    if (aidOffers !== prev.aidOffers) log('mutual-aid', `Aid offers: ${prev.aidOffers} → ${aidOffers}`);
    if (aidRequests !== prev.aidRequests) log('mutual-aid', `Aid requests: ${prev.aidRequests} → ${aidRequests}`);
  }

  const freshness = createFreshness(fetchData, 30_000);
  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } = freshness;

  onMount(async () => {
    await refresh();
    loading = false;
    startPolling();
  });

  onDestroy(() => stopPolling());

  // ============================================================================
  // Helpers
  // ============================================================================

  function tierColor(tier: string): string {
    switch (tier) {
      case 'Normal': return 'text-green-400';
      case 'Elevated': return 'text-yellow-400';
      case 'High': return 'text-orange-400';
      case 'Emergency': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }

  function domainColor(domain: string): string {
    switch (domain) {
      case 'tend': return 'text-blue-400';
      case 'mutual-aid': return 'text-green-400';
      case 'water': return 'text-cyan-400';
      case 'emergency': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }

  function formatLogTime(ts: number): string {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  // ============================================================================
  // Community Config (localStorage-backed)
  // ============================================================================

  const CONFIG_KEY = 'mycelix-community-config';

  interface CommunityConfig {
    name: string;
    location: string;
    contactName: string;
    contactPhone: string;
    contactEmail: string;
  }

  const communityDefaults = getCommunityConfig();
  const defaultConfig: CommunityConfig = {
    name: communityDefaults.community_name,
    location: '',
    contactName: '',
    contactPhone: '',
    contactEmail: '',
  };

  let communityConfig: CommunityConfig = loadConfig();
  let showConfigForm = false;
  let configSaved = false;

  function loadConfig(): CommunityConfig {
    if (!browser) return { ...defaultConfig };
    try {
      const raw = localStorage.getItem(CONFIG_KEY);
      if (!raw) return { ...defaultConfig };
      return { ...defaultConfig, ...JSON.parse(raw) };
    } catch {
      return { ...defaultConfig };
    }
  }

  function saveConfig() {
    if (!browser) return;
    localStorage.setItem(CONFIG_KEY, JSON.stringify(communityConfig));
    configSaved = true;
    setTimeout(() => { configSaved = false; }, 2000);
  }

  // ============================================================================
  // Config Export/Import
  // ============================================================================

  let configImportError = '';
  let configImportSuccess = false;

  function exportConfig() {
    const config = {
      community_name: communityConfig.name,
      operator: {
        name: communityConfig.contactName,
        phone: communityConfig.contactPhone,
        email: communityConfig.contactEmail,
        location: communityConfig.location,
      },
      community_config: communityDefaults,
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mycelix-config-${communityConfig.name.toLowerCase().replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importConfig(event: Event) {
    configImportError = '';
    configImportSuccess = false;
    const input = event.target as HTMLInputElement;
    const file = input?.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        if (!data.community_name && !data.community_config?.community_name) {
          configImportError = 'Invalid config: missing community_name';
          return;
        }
        // Apply operator config
        if (data.operator) {
          communityConfig.contactName = data.operator.name || '';
          communityConfig.contactPhone = data.operator.phone || '';
          communityConfig.contactEmail = data.operator.email || '';
          communityConfig.location = data.operator.location || '';
        }
        if (data.community_name) {
          communityConfig.name = data.community_name;
        }
        saveConfig();
        configImportSuccess = true;
      } catch {
        configImportError = 'Failed to parse JSON file';
      }
    };
    reader.readAsText(file);
    // Reset input so same file can be re-imported
    input.value = '';
  }
</script>

<svelte:head>
  <title>Operator Dashboard | Mycelix Observatory</title>
</svelte:head>

{#if loading}
  <div class="text-white p-8 text-center text-gray-400">Loading operator dashboard...</div>
{:else}
<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div>
        <h1 class="text-lg font-bold">Operator Dashboard</h1>
        <p class="text-xs text-gray-400">System health and community activity at a glance</p>
      </div>
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-1.5 text-sm">
          <span class="relative flex h-2.5 w-2.5" aria-label="Connection quality: {$connectionQuality}">
            <span class="relative inline-flex rounded-full h-2.5 w-2.5 {$qualityColor}"></span>
          </span>
          <span class="text-gray-300">{$connectionLabel}</span>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />

    <!-- System Health -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700" role="status">
        <h3 class="text-gray-400 text-xs uppercase">Connection</h3>
        <p class="text-xl font-bold mt-1 {$connectionQuality === 'excellent' ? 'text-green-400' : $connectionQuality === 'degraded' ? 'text-yellow-400' : 'text-red-400'}">
          {$connectionQuality === 'excellent' ? 'Healthy' : $connectionQuality === 'degraded' ? 'Degraded' : 'Down'}
        </p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700" role="status">
        <h3 class="text-gray-400 text-xs uppercase">Oracle Tier</h3>
        <p class="text-xl font-bold mt-1 {tierColor(oracle.tier)}">{oracle.tier}</p>
        <p class="text-xs text-gray-500 mt-1">Vitality: {oracle.vitality}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Queued Offline</h3>
        <p class="text-xl font-bold mt-1 {$queueCount > 0 ? 'text-amber-400' : 'text-green-400'}">{$queueCount}</p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Water Alerts</h3>
        <p class="text-xl font-bold mt-1 {waterAlerts > 0 ? 'text-red-400' : 'text-green-400'}">{waterAlerts}</p>
      </div>
    </div>

    <!-- Domain Counts -->
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 mb-6">
      <a href="/tend" aria-label="View TEND listings ({tendListings})" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-blue-500/50 transition-colors">
        <p class="text-xs text-gray-400">TEND Listings</p>
        <p class="text-2xl font-bold text-blue-400">{tendListings}</p>
      </a>
      <a href="/tend" aria-label="View TEND requests ({tendRequests})" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-blue-500/50 transition-colors">
        <p class="text-xs text-gray-400">TEND Requests</p>
        <p class="text-2xl font-bold text-blue-300">{tendRequests}</p>
      </a>
      <a href="/mutual-aid" aria-label="View mutual aid offers ({aidOffers})" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-green-500/50 transition-colors">
        <p class="text-xs text-gray-400">Aid Offers</p>
        <p class="text-2xl font-bold text-green-400">{aidOffers}</p>
      </a>
      <a href="/mutual-aid" aria-label="View mutual aid requests ({aidRequests})" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-green-500/50 transition-colors">
        <p class="text-xs text-gray-400">Aid Requests</p>
        <p class="text-2xl font-bold text-orange-400">{aidRequests}</p>
      </a>
      <a href="/emergency" aria-label="View emergency channels ({emergencyChannels})" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-red-500/50 transition-colors">
        <p class="text-xs text-gray-400">Emerg. Channels</p>
        <p class="text-2xl font-bold text-red-400">{emergencyChannels}</p>
      </a>
      <a href="/food" aria-label="View food plots ({foodPlots})" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-emerald-500/50 transition-colors">
        <p class="text-xs text-gray-400">Food Plots</p>
        <p class="text-2xl font-bold text-emerald-400">{foodPlots}</p>
      </a>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Low Stock Alerts -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-sm font-semibold">Low Stock Alerts</h2>
          <a href="/supplies" aria-label="View all supply items" class="text-xs text-blue-400 hover:text-blue-300">View all</a>
        </div>
        <div class="p-4 space-y-2 max-h-64 overflow-y-auto">
          {#each lowStockItems as ls}
            <div class="flex justify-between items-center p-2 bg-yellow-900/20 border border-yellow-800/50 rounded">
              <div>
                <p class="text-sm font-medium">{ls.item.name}</p>
                <p class="text-xs text-gray-400">{ls.item.category}</p>
              </div>
              <div class="text-right">
                <p class="text-sm font-bold text-yellow-400">{ls.total_stock} {ls.item.unit}</p>
                <p class="text-xs text-gray-500">min: {ls.item.reorder_point}</p>
              </div>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-4 text-sm">All stock levels OK</p>
          {/each}
        </div>
      </div>

      <!-- Activity Log -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-sm font-semibold">Activity Log</h2>
        </div>
        <div class="p-4 space-y-1 max-h-64 overflow-y-auto" aria-live="polite">
          {#each activityLog as entry}
            <div class="flex items-start gap-2 text-xs py-1">
              <span class="text-gray-500 font-mono whitespace-nowrap">{formatLogTime(entry.time)}</span>
              <span class="font-medium {domainColor(entry.domain)}">[{entry.domain}]</span>
              <span class="text-gray-300">{entry.message}</span>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-4 text-sm">No activity yet — changes will appear here as data updates</p>
          {/each}
        </div>
      </div>

      <!-- Offline Queue Inspector -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 lg:col-span-2">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-sm font-semibold">Offline Queue <span class="text-gray-400 font-normal">({queueItems.length} item{queueItems.length !== 1 ? 's' : ''})</span></h2>
          {#if queueItems.length > 0}
            <button
              class="text-xs text-red-400 hover:text-red-300 transition-colors"
              on:click={async () => {
                if (confirm('Clear all queued submissions? This cannot be undone.')) {
                  await clearQueue();
                  queueItems = await getQueue();
                }
              }}
            >Clear All</button>
          {/if}
        </div>
        <div class="p-4 space-y-2 max-h-72 overflow-y-auto">
          {#each queueItems as item (item.id)}
            <div class="flex items-center justify-between p-2 bg-gray-700/40 border border-gray-600/50 rounded">
              <div class="flex items-center gap-3 min-w-0">
                <div class="min-w-0">
                  <p class="text-sm font-medium truncate">{item.domain} <span class="text-gray-400 font-normal">/ {item.action}</span></p>
                  <p class="text-xs text-gray-500">{formatLogTime(item.created_at)} &middot; {item.attempts} attempt{item.attempts !== 1 ? 's' : ''}</p>
                </div>
              </div>
              <div class="flex items-center gap-2 shrink-0">
                <span class="text-xs px-1.5 py-0.5 rounded {item.status === 'queued' ? 'bg-yellow-900/40 text-yellow-400' : item.status === 'sending' ? 'bg-blue-900/40 text-blue-400' : 'bg-red-900/40 text-red-400'}">{item.status}</span>
                <button
                  class="text-xs text-gray-400 hover:text-red-400 transition-colors"
                  on:click={async () => {
                    await removeItem(item.id);
                    queueItems = await getQueue();
                  }}
                >Remove</button>
              </div>
            </div>
          {:else}
            <p class="text-gray-500 text-center py-4 text-sm">Queue empty — all submissions synced</p>
          {/each}
        </div>
      </div>
    </div>

    <!-- Community Config -->
    <div class="mt-6 bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700 flex justify-between items-center">
        <h2 class="text-sm font-semibold">Community Identity</h2>
        <button
          on:click={() => showConfigForm = !showConfigForm}
          class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          aria-label={showConfigForm ? 'Close community settings' : 'Edit community settings'}
        >
          {showConfigForm ? 'Close' : 'Edit'}
        </button>
      </div>
      {#if showConfigForm}
        <div class="p-4">
          <form on:submit|preventDefault={saveConfig} class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label for="cfg-name" class="text-xs text-gray-400">Community Name</label>
              <input id="cfg-name" bind:value={communityConfig.name}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
            </div>
            <div>
              <label for="cfg-location" class="text-xs text-gray-400">Location</label>
              <input id="cfg-location" bind:value={communityConfig.location}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
            </div>
            <div>
              <label for="cfg-contact" class="text-xs text-gray-400">Operator Name</label>
              <input id="cfg-contact" bind:value={communityConfig.contactName}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
            </div>
            <div>
              <label for="cfg-phone" class="text-xs text-gray-400">Operator Phone</label>
              <input id="cfg-phone" type="tel" bind:value={communityConfig.contactPhone} placeholder="+27 82 000 0000"
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
            </div>
            <div class="sm:col-span-2">
              <label for="cfg-email" class="text-xs text-gray-400">Operator Email</label>
              <input id="cfg-email" type="email" bind:value={communityConfig.contactEmail}
                class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
            </div>
            <div class="sm:col-span-2 flex items-center gap-3">
              <button type="submit"
                class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm font-medium transition-colors">
                Save
              </button>
              {#if configSaved}
                <span class="text-xs text-green-400">Saved</span>
              {/if}
            </div>
          </form>
        </div>
      {:else}
        <div class="p-4 grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
          <div>
            <p class="text-xs text-gray-400">Community</p>
            <p class="text-gray-200">{communityConfig.name}</p>
          </div>
          <div>
            <p class="text-xs text-gray-400">Location</p>
            <p class="text-gray-200">{communityConfig.location}</p>
          </div>
          <div>
            <p class="text-xs text-gray-400">Operator</p>
            <p class="text-gray-200">{communityConfig.contactName || 'Not set'}</p>
          </div>
          <div>
            <p class="text-xs text-gray-400">Contact</p>
            <p class="text-gray-200">{communityConfig.contactPhone || communityConfig.contactEmail || 'Not set'}</p>
          </div>
        </div>
      {/if}
    </div>

    <!-- Config Export/Import -->
    <div class="mt-6 bg-gray-800 rounded-lg border border-gray-700">
      <div class="p-4 border-b border-gray-700 flex justify-between items-center">
        <h2 class="text-sm font-semibold">Community Config (JSON)</h2>
        <div class="flex gap-2">
          <button
            on:click={exportConfig}
            class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >Export</button>
          <label class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors cursor-pointer">
            Import
            <input type="file" accept=".json" class="hidden" on:change={importConfig} />
          </label>
        </div>
      </div>
      {#if configImportError}
        <div class="p-3 bg-red-900/30 border-b border-red-500/50 text-sm text-red-300">{configImportError}</div>
      {/if}
      {#if configImportSuccess}
        <div class="p-3 bg-green-900/30 border-b border-green-500/50 text-sm text-green-300">Config imported. Reload the page for changes to take effect.</div>
      {/if}
    </div>

    <!-- Quick Links -->
    <div class="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
      <a href="/print" aria-label="Print summary — printable community report" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Print Summary</p>
        <p class="text-xs text-gray-500 mt-1">Printable community report</p>
      </a>
      <a href="/value-anchor" aria-label="Price oracle — update local prices" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Price Oracle</p>
        <p class="text-xs text-gray-500 mt-1">Update local prices</p>
      </a>
      <a href="/water" aria-label="Water safety — {waterSystems} systems, {waterAlerts} alerts" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Water Safety</p>
        <p class="text-xs text-gray-500 mt-1">{waterSystems} systems, {waterAlerts} alerts</p>
      </a>
      <a href="/network" aria-label="Network health — mesh bridge status" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Network / Mesh</p>
        <p class="text-xs text-gray-500 mt-1">DHT + mesh bridge status</p>
      </a>
    </div>

    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Operator Dashboard &middot; Mycelix Resilience Kit</p>
    </footer>
  </main>
</div>
{/if}
