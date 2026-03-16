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
  import { queueCount } from '$lib/offline-queue';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';

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
          <span class="relative flex h-2.5 w-2.5">
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
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 class="text-gray-400 text-xs uppercase">Connection</h3>
        <p class="text-xl font-bold mt-1 {$connectionQuality === 'excellent' ? 'text-green-400' : $connectionQuality === 'degraded' ? 'text-yellow-400' : 'text-red-400'}">
          {$connectionQuality === 'excellent' ? 'Healthy' : $connectionQuality === 'degraded' ? 'Degraded' : 'Down'}
        </p>
      </div>
      <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
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
      <a href="/tend" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-blue-500/50 transition-colors">
        <p class="text-xs text-gray-400">TEND Listings</p>
        <p class="text-2xl font-bold text-blue-400">{tendListings}</p>
      </a>
      <a href="/tend" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-blue-500/50 transition-colors">
        <p class="text-xs text-gray-400">TEND Requests</p>
        <p class="text-2xl font-bold text-blue-300">{tendRequests}</p>
      </a>
      <a href="/mutual-aid" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-green-500/50 transition-colors">
        <p class="text-xs text-gray-400">Aid Offers</p>
        <p class="text-2xl font-bold text-green-400">{aidOffers}</p>
      </a>
      <a href="/mutual-aid" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-green-500/50 transition-colors">
        <p class="text-xs text-gray-400">Aid Requests</p>
        <p class="text-2xl font-bold text-orange-400">{aidRequests}</p>
      </a>
      <a href="/emergency" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-red-500/50 transition-colors">
        <p class="text-xs text-gray-400">Emerg. Channels</p>
        <p class="text-2xl font-bold text-red-400">{emergencyChannels}</p>
      </a>
      <a href="/food" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-emerald-500/50 transition-colors">
        <p class="text-xs text-gray-400">Food Plots</p>
        <p class="text-2xl font-bold text-emerald-400">{foodPlots}</p>
      </a>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Low Stock Alerts -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-sm font-semibold">Low Stock Alerts</h2>
          <a href="/supplies" class="text-xs text-blue-400 hover:text-blue-300">View all</a>
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
        <div class="p-4 space-y-1 max-h-64 overflow-y-auto">
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
    </div>

    <!-- Quick Links -->
    <div class="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
      <a href="/print" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Print Summary</p>
        <p class="text-xs text-gray-500 mt-1">Printable community report</p>
      </a>
      <a href="/value-anchor" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Price Oracle</p>
        <p class="text-xs text-gray-500 mt-1">Update local prices</p>
      </a>
      <a href="/water" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Water Safety</p>
        <p class="text-xs text-gray-500 mt-1">{waterSystems} systems, {waterAlerts} alerts</p>
      </a>
      <a href="/household" class="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-500 transition-colors text-center">
        <p class="text-sm font-medium text-gray-300">Households</p>
        <p class="text-xs text-gray-500 mt-1">Emergency plans</p>
      </a>
    </div>

    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Operator Dashboard &middot; Mycelix Resilience Kit</p>
    </footer>
  </main>
</div>
{/if}
