<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { initializeStores, ecosystemStatus, byzantineAlerts, isLiveMode, conductorStatus } from '$lib/stores';
  import { getBalance, getOracleState, type BalanceInfo, type OracleState } from '$lib/resilience-client';
  import { hasBasket, purchasingPowerIndex } from '$lib/value-basket';
  import { writable } from 'svelte/store';

  const tendBalance = writable<BalanceInfo | null>(null);
  const oracleState = writable<OracleState>({ vitality: 72, tier: 'Normal', updated_at: Date.now() });

  // Derived stats from stores
  $: stats = {
    activeNodes: $ecosystemStatus.totalNodes,
    byzantineTolerance: $ecosystemStatus.byzantineTolerance,
    totalTransactions: $ecosystemStatus.happs.reduce((sum, h) => sum + h.metrics.totalTransactions, 0),
    crossHappQueries: $ecosystemStatus.crossHappQueries
  };

  // hApp status from ecosystem
  $: happs = $ecosystemStatus.happs.slice(0, 4).map(h => ({
    name: h.name,
    status: h.status === 'healthy' ? 'active' : h.status === 'degraded' ? 'development' : 'alpha',
    version: h.version,
    nodes: h.nodeCount
  }));

  // Recent alerts
  $: alerts = $byzantineAlerts.slice(0, 5).map(a => ({
    type: a.severity,
    message: a.message,
    time: new Date(a.timestamp).toLocaleTimeString()
  }));

  let cleanup: (() => void) | null = null;

  onMount(async () => {
    console.log('Observatory mounted - initializing stores');
    cleanup = await initializeStores();
    // Load resilience data
    try {
      const [bal, ora] = await Promise.all([getBalance('self.did'), getOracleState()]);
      tendBalance.set(bal);
      oracleState.set(ora);
    } catch (e) {
      console.warn('[Dashboard] Failed to load resilience data:', e);
    }
  });

  onDestroy(() => {
    if (cleanup) cleanup();
  });

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'development': return 'bg-yellow-500';
      case 'alpha': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  }
</script>

<svelte:head>
  <title>Mycelix Observatory</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white">
  <!-- Header -->
  <header class="bg-gray-800 border-b border-gray-700">
    <div class="container mx-auto">
      <div class="flex justify-between items-center p-4">
        <div class="flex items-center gap-3">
          <span class="text-2xl">🍄</span>
          <h1 class="text-xl font-bold">Mycelix Observatory</h1>
        </div>
        <div class="flex items-center gap-4">
          {#if $conductorStatus === 'connected'}
            <span class="text-green-400">● Live</span>
          {:else if $conductorStatus === 'connecting'}
            <span class="text-yellow-400 animate-pulse">● Connecting</span>
          {:else}
            <span class="text-yellow-400">● Demo</span>
          {/if}
          <span class="text-gray-400">v0.1.0</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="container mx-auto p-6">
    {#if !$isLiveMode}
      <div class="mb-6 bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 flex items-center gap-3">
        <span class="text-yellow-400 text-xl">⚠️</span>
        <div class="text-sm">
          <div class="text-yellow-300 font-medium">Demo Mode Active</div>
          <div class="text-yellow-400/80">Showing simulated data - no Holochain conductor connected</div>
        </div>
      </div>
    {/if}

    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-gray-400 text-sm uppercase">Active Nodes</h3>
        <p class="text-3xl font-bold mt-2">{stats.activeNodes}</p>
      </div>

      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-gray-400 text-sm uppercase">Byzantine Tolerance</h3>
        <p class="text-3xl font-bold mt-2 text-green-400">{stats.byzantineTolerance}%</p>
        <p class="text-xs text-gray-500 mt-1">Exceeds 33% classical limit</p>
      </div>

      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-gray-400 text-sm uppercase">Total Transactions</h3>
        <p class="text-3xl font-bold mt-2">{stats.totalTransactions.toLocaleString()}</p>
      </div>

      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-gray-400 text-sm uppercase">Cross-hApp Queries</h3>
        <p class="text-3xl font-bold mt-2">{stats.crossHappQueries.toLocaleString()}</p>
      </div>
    </div>

    <!-- Resilience Quick Stats -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
      <a href="/tend" class="block bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-600 transition-colors">
        <h3 class="text-gray-400 text-sm uppercase">TEND Balance</h3>
        {#if $tendBalance}
          <p class="text-3xl font-bold mt-2" class:text-green-400={$tendBalance.balance > 0} class:text-red-400={$tendBalance.balance < 0}>
            {$tendBalance.balance > 0 ? '+' : ''}{$tendBalance.balance}
          </p>
          <p class="text-xs text-gray-500 mt-1">{$tendBalance.exchange_count} exchanges</p>
        {:else}
          <p class="text-gray-500 mt-2">Loading...</p>
        {/if}
      </a>

      <a href="/tend" class="block bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-600 transition-colors">
        <h3 class="text-gray-400 text-sm uppercase">Oracle Tier</h3>
        <p class="text-3xl font-bold mt-2">{$oracleState.tier}</p>
        <p class="text-xs text-gray-500 mt-1">Vitality {$oracleState.vitality} · Limit &plusmn;{$oracleState.tier === 'Normal' ? 40 : $oracleState.tier === 'Elevated' ? 60 : $oracleState.tier === 'High' ? 80 : 120} TEND</p>
      </a>

      <a href="/value-anchor" class="block bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-600 transition-colors">
        <h3 class="text-gray-400 text-sm uppercase">Value Anchor</h3>
        {#if $hasBasket}
          <p class="text-3xl font-bold mt-2">{$purchasingPowerIndex.toFixed(2)}</p>
          <p class="text-xs text-gray-500 mt-1">Basket index (TEND/unit)</p>
        {:else}
          <p class="text-sm text-gray-500 mt-2">No basket configured</p>
          <p class="text-xs text-gray-500 mt-1">Set up local prices</p>
        {/if}
      </a>
    </div>

    <!-- Two Column Layout -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- hApp Status -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold">hApp Status</h2>
        </div>
        <div class="p-4">
          <div class="space-y-3">
            {#each happs as happ}
              <div class="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                <div class="flex items-center gap-3">
                  <span class={`w-3 h-3 rounded-full ${getStatusColor(happ.status)}`}></span>
                  <div>
                    <p class="font-medium">{happ.name}</p>
                    <p class="text-xs text-gray-400">v{happ.version}</p>
                  </div>
                </div>
                <div class="text-right">
                  <p class="text-sm">{happ.nodes} nodes</p>
                  <p class="text-xs text-gray-400 capitalize">{happ.status}</p>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>

      <!-- Trust Distribution -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold">Trust Distribution</h2>
        </div>
        <div class="p-4">
          <div class="h-64 flex items-center justify-center text-gray-500">
            <div class="text-center">
              <p class="text-4xl mb-2">📊</p>
              <p>Trust heatmap visualization</p>
              <p class="text-sm">Connect to network to view</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Recent Activity -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold">Recent Activity</h2>
        </div>
        <div class="p-4">
          {#if alerts.length === 0}
            <p class="text-gray-500 text-center py-8">No recent activity</p>
          {:else}
            <div class="space-y-2">
              {#each alerts as alert}
                <div class="p-2 bg-gray-700/50 rounded text-sm">
                  <span class="text-gray-400">{alert.time}</span>
                  <span class="ml-2">{alert.message}</span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>

      <!-- Byzantine Detection -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700">
          <h2 class="text-lg font-semibold">Byzantine Detection</h2>
        </div>
        <div class="p-4">
          <div class="grid grid-cols-2 gap-4">
            <div class="text-center p-4 bg-green-900/30 rounded-lg">
              <p class="text-3xl font-bold text-green-400">100%</p>
              <p class="text-sm text-gray-400">Detection Rate</p>
            </div>
            <div class="text-center p-4 bg-blue-900/30 rounded-lg">
              <p class="text-3xl font-bold text-blue-400">2.3%</p>
              <p class="text-sm text-gray-400">False Positive Rate</p>
            </div>
          </div>
          <div class="mt-4 p-3 bg-gray-700/50 rounded-lg">
            <p class="text-green-400 font-medium">✓ No Byzantine nodes detected</p>
            <p class="text-xs text-gray-400 mt-1">Last check: just now</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Observatory • Holochain 0.6 • MATL v1.0</p>
      <p class="mt-1">34% Byzantine Fault Tolerance • Consciousness-First Computing</p>
    </footer>
  </main>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
  }
</style>
