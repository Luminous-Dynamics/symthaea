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
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';
  import { conductorStatus } from '$lib/stores';
  import {
    getBalance,
    getAllPlots,
    getChannels,
    getAllWaterSystems,
    getActiveWaterAlerts,
    getMyHearths,
    getAllCareCircles,
    getAvailableUnits,
    getAllInventoryItems,
    getLowStockItems,
    getGraphStats,
  } from '$lib/resilience-client';

  type CardStatus = 'loading' | 'ok' | 'error';

  interface DomainCard {
    domain: string;
    metric: string;
    color: string;
    borderColor: string;
    textColor: string;
    link: string;
    status: CardStatus;
  }

  import { getDefaultDao } from '$lib/community';

  const DEFAULT_MEMBER = 'did:key:local-operator';
  const DEFAULT_DAO = getDefaultDao();

  let cards: DomainCard[] = [
    { domain: 'TEND', metric: '--', color: 'bg-green-500', borderColor: 'border-green-700', textColor: 'text-green-400', link: '/tend', status: 'loading' },
    { domain: 'Food', metric: '--', color: 'bg-green-500', borderColor: 'border-green-700', textColor: 'text-green-400', link: '/food', status: 'loading' },
    { domain: 'Mutual Aid', metric: '--', color: 'bg-blue-500', borderColor: 'border-blue-700', textColor: 'text-blue-400', link: '/mutual-aid', status: 'loading' },
    { domain: 'Emergency', metric: '--', color: 'bg-red-500', borderColor: 'border-red-700', textColor: 'text-red-400', link: '/emergency', status: 'loading' },
    { domain: 'Value Anchor', metric: '--', color: 'bg-amber-500', borderColor: 'border-amber-700', textColor: 'text-amber-400', link: '/value-anchor', status: 'loading' },
    { domain: 'Water', metric: '--', color: 'bg-cyan-500', borderColor: 'border-cyan-700', textColor: 'text-cyan-400', link: '/water', status: 'loading' },
    { domain: 'Household', metric: '--', color: 'bg-indigo-500', borderColor: 'border-indigo-700', textColor: 'text-indigo-400', link: '/household', status: 'loading' },
    { domain: 'Knowledge', metric: '--', color: 'bg-purple-500', borderColor: 'border-purple-700', textColor: 'text-purple-400', link: '/knowledge', status: 'loading' },
    { domain: 'Care Circles', metric: '--', color: 'bg-pink-500', borderColor: 'border-pink-700', textColor: 'text-pink-400', link: '/care-circles', status: 'loading' },
    { domain: 'Shelter', metric: '--', color: 'bg-emerald-500', borderColor: 'border-emerald-700', textColor: 'text-emerald-400', link: '/shelter', status: 'loading' },
    { domain: 'Supplies', metric: '--', color: 'bg-amber-500', borderColor: 'border-amber-700', textColor: 'text-amber-400', link: '/supplies', status: 'loading' },
  ];

  function updateCard(index: number, metric: string, status: CardStatus) {
    cards[index] = { ...cards[index], metric, status };
    cards = cards;
  }

  async function fetchData(): Promise<void> {
    const fetchers: Array<() => Promise<void>> = [
      // 0: TEND — Balance
      async () => {
        try {
          const bal = await getBalance(DEFAULT_MEMBER, DEFAULT_DAO);
          updateCard(0, `${bal.balance.toFixed(1)} TEND`, 'ok');
        } catch { updateCard(0, 'N/A', 'error'); }
      },
      // 1: Food — Active Plots
      async () => {
        try {
          const plots = await getAllPlots();
          updateCard(1, `${plots.length} Plots`, 'ok');
        } catch { updateCard(1, 'N/A', 'error'); }
      },
      // 2: Mutual Aid — static
      async () => {
        try {
          updateCard(2, 'Active', 'ok');
        } catch { updateCard(2, 'N/A', 'error'); }
      },
      // 3: Emergency — Channels
      async () => {
        try {
          const ch = await getChannels();
          updateCard(3, `${ch.length} Channels`, 'ok');
        } catch { updateCard(3, 'N/A', 'error'); }
      },
      // 4: Value Anchor — static
      async () => {
        try {
          updateCard(4, 'Configured', 'ok');
        } catch { updateCard(4, 'N/A', 'error'); }
      },
      // 5: Water — Systems / Alerts
      async () => {
        try {
          const [systems, alerts] = await Promise.all([
            getAllWaterSystems(),
            getActiveWaterAlerts(),
          ]);
          updateCard(5, `${systems.length} Sys / ${alerts.length} Alerts`, 'ok');
        } catch { updateCard(5, 'N/A', 'error'); }
      },
      // 6: Household — Hearths
      async () => {
        try {
          const hearths = await getMyHearths();
          updateCard(6, `${hearths.length} Hearths`, 'ok');
        } catch { updateCard(6, 'N/A', 'error'); }
      },
      // 7: Knowledge — Claims (graph stats)
      async () => {
        try {
          const stats = await getGraphStats();
          updateCard(7, `${stats.total_claims} Claims`, 'ok');
        } catch { updateCard(7, 'N/A', 'error'); }
      },
      // 8: Care Circles
      async () => {
        try {
          const circles = await getAllCareCircles();
          updateCard(8, `${circles.length} Circles`, 'ok');
        } catch { updateCard(8, 'N/A', 'error'); }
      },
      // 9: Shelter — Available Units
      async () => {
        try {
          const units = await getAvailableUnits();
          updateCard(9, `${units.length} Units`, 'ok');
        } catch { updateCard(9, 'N/A', 'error'); }
      },
      // 10: Supplies — Items / Low Stock
      async () => {
        try {
          const [items, low] = await Promise.all([
            getAllInventoryItems(),
            getLowStockItems(),
          ]);
          updateCard(10, `${items.length} Items / ${low.length} Low`, 'ok');
        } catch { updateCard(10, 'N/A', 'error'); }
      },
    ];

    await Promise.all(fetchers.map((fn) => fn()));
  }

  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } =
    createFreshness(fetchData, 120_000);

  onMount(async () => {
    await refresh();
    startPolling();
  });

  onDestroy(() => stopPolling());

  function loadCommunityName(): string {
    if (!browser) return 'Mycelix Community';
    try {
      const raw = localStorage.getItem('mycelix-community-config');
      if (raw) {
        const cfg = JSON.parse(raw);
        if (cfg.name) return cfg.name;
      }
    } catch { /* ignore */ }
    return 'Mycelix Community';
  }

  const communityName = loadCommunityName();

  function statusDotColor(status: CardStatus): string {
    switch (status) {
      case 'ok': return 'bg-green-400';
      case 'loading': return 'bg-yellow-400 animate-pulse';
      case 'error': return 'bg-red-400';
    }
  }
</script>

<svelte:head>
  <title>Resilience Kit — Mycelix Observatory</title>
</svelte:head>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <!-- Header -->
  <div class="max-w-6xl mx-auto mb-8">
    <h1 class="text-3xl font-bold text-white">Mycelix Resilience Kit</h1>
    <p class="text-gray-400 mt-1">{communityName}</p>
  </div>

  <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />

  <!-- Domain Grid -->
  <div class="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-10">
    {#each cards as card}
      <a
        href={card.link}
        class="block bg-gray-900 border border-gray-800 rounded-lg p-4 h-[120px] flex flex-col justify-between
               hover:border-gray-600 hover:bg-gray-800/60 transition-colors"
      >
        <div class="flex items-center justify-between">
          <span class="text-sm font-medium text-gray-400">{card.domain}</span>
          <span class="w-2.5 h-2.5 rounded-full {statusDotColor(card.status)}" aria-hidden="true"></span>
        </div>
        <div class="mt-auto">
          <span class="text-2xl font-bold {card.textColor}">{card.metric}</span>
        </div>
      </a>
    {/each}
  </div>

  <!-- System Status -->
  <div class="max-w-6xl mx-auto">
    <h2 class="text-lg font-semibold text-gray-300 mb-3">System Status</h2>
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-4 grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full {
          $conductorStatus === 'connected' ? 'bg-green-400' :
          $conductorStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' :
          $conductorStatus === 'demo' ? 'bg-yellow-400' :
          'bg-red-400'
        }" aria-hidden="true"></span>
        <span class="text-gray-400">Conductor:</span>
        <span class="text-gray-200 capitalize">{$conductorStatus}</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-yellow-400" aria-hidden="true"></span>
        <span class="text-gray-400">Mesh Bridge:</span>
        <span class="text-gray-200">Standby</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-gray-400">Last Updated:</span>
        <span class="text-gray-200">{$lastUpdated ? new Date($lastUpdated).toLocaleTimeString() : 'Fetching...'}</span>
      </div>
    </div>
  </div>
</div>
