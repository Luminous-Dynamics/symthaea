<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
--><script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { conductorStatus } from '$lib/stores';
  import { connectToConductor, flushOfflineQueue } from '$lib/conductor';
  import { queueCount, initQueueCount } from '$lib/offline-queue';
  import { connectionQuality, connectionLabel, qualityColor, setInternet, setMeshBridgeDetected } from '$lib/connection-health';
  import { isMeshBridgeAvailable } from '$lib/offline-queue';
  import { refreshConsensus } from '$lib/value-basket';
  import Toast from '$lib/components/Toast.svelte';
  import LocaleSelector from '$lib/components/LocaleSelector.svelte';

  let syncing = false;

  async function handleSyncNow() {
    if (syncing) return;
    syncing = true;
    try {
      await flushOfflineQueue();
    } finally {
      syncing = false;
    }
  }

  let isOnline = true;
  let mobileMenuOpen = false;
  let consensusTimer: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    isOnline = navigator.onLine;
    setInternet(isOnline);
    const goOnline = () => { isOnline = true; setInternet(true); };
    const goOffline = () => { isOnline = false; setInternet(false); };
    window.addEventListener('online', goOnline);
    window.addEventListener('offline', goOffline);
    initQueueCount();

    // Proactive conductor connection on app startup
    connectToConductor();

    // Detect mesh bridge availability (non-blocking)
    isMeshBridgeAvailable().then(setMeshBridgeDetected).catch(() => {});
    // Re-check mesh bridge every 60 seconds
    const meshCheckTimer = setInterval(() => {
      isMeshBridgeAvailable().then(setMeshBridgeDetected).catch(() => {});
    }, 60_000);

    // Register service worker for offline PWA support
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/service-worker.js').catch((err) =>
        console.warn('[Observatory] Service worker registration failed:', err),
      );
    }

    // Global consensus refresh every 5 minutes
    refreshConsensus().catch(() => {});
    consensusTimer = setInterval(() => {
      refreshConsensus().catch(() => {});
    }, 300_000);

    return () => {
      window.removeEventListener('online', goOnline);
      window.removeEventListener('offline', goOffline);
      if (consensusTimer) clearInterval(consensusTimer);
      clearInterval(meshCheckTimer);
    };
  });

  $: statusColor = {
    connecting: 'bg-yellow-500',
    connected: 'bg-green-500',
    disconnected: 'bg-red-500',
    demo: 'bg-yellow-500',
  }[$conductorStatus] ?? 'bg-gray-500';

  $: currentPath = $page.url.pathname;

  function isActive(href: string): boolean {
    if (href === '/') return currentPath === '/';
    return currentPath === href || currentPath.startsWith(href + '/');
  }

  function linkClass(href: string): string {
    const base = 'px-3 py-1.5 rounded transition-colors whitespace-nowrap';
    return isActive(href)
      ? `${base} bg-gray-700 text-white font-medium`
      : `${base} text-gray-300 hover:bg-gray-800 hover:text-white`;
  }

  type NavItem = { href: string; label: string };
  type NavGroup = { label: string; items: NavItem[] };

  const navGroups: NavGroup[] = [
    {
      label: 'Overview',
      items: [
        { href: '/', label: 'Dashboard' },
        { href: '/resilience', label: 'Resilience' },
      ],
    },
    {
      label: 'Resilience Kit',
      items: [
        { href: '/tend', label: 'TEND' },
        { href: '/food', label: 'Food' },
        { href: '/mutual-aid', label: 'Mutual Aid' },
        { href: '/emergency', label: 'Emergency' },
        { href: '/value-anchor', label: 'Value Anchor' },
        { href: '/water', label: 'Water' },
        { href: '/household', label: 'Household' },
        { href: '/knowledge', label: 'Knowledge' },
        { href: '/care-circles', label: 'Care Circles' },
        { href: '/shelter', label: 'Shelter' },
        { href: '/supplies', label: 'Supplies' },
      ],
    },
    {
      label: 'Observatory',
      items: [
        { href: '/admin', label: 'Operator' },
        { href: '/governance', label: 'Governance' },
        { href: '/budgets', label: 'Budgets' },
        { href: '/sagas', label: 'Workflows' },
        { href: '/notifications', label: 'Notifications' },
        { href: '/network', label: 'Network' },
        { href: '/analytics', label: 'Analytics' },
        { href: '/attribution', label: 'Attribution' },
      ],
    },
  ];

  // Close mobile menu on navigation
  $: if (currentPath) mobileMenuOpen = false;
</script>

<!-- Network offline banner — highest priority -->
{#if !isOnline}
  <div class="fixed bottom-0 left-0 right-0 z-50 px-4 py-2 text-sm flex items-center justify-center gap-2 bg-orange-900/95 text-orange-100">
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 5.636a9 9 0 010 12.728M5.636 5.636a9 9 0 000 12.728M12 12h.01" />
    </svg>
    <span>Offline — no internet connection. Cached data shown. Mesh bridge continues operating if running locally.</span>
  </div>
<!-- Conductor status bar — visible when not connected -->
{:else if $conductorStatus !== 'connected'}
  <div class="fixed bottom-0 left-0 right-0 z-50 px-4 py-2 text-sm flex items-center justify-center gap-2
    {$conductorStatus === 'demo' ? 'bg-yellow-900/90 text-yellow-200' : ''}
    {$conductorStatus === 'connecting' ? 'bg-yellow-900/90 text-yellow-200' : ''}
    {$conductorStatus === 'disconnected' ? 'bg-red-900/90 text-red-200' : ''}">
    <span class="relative flex h-2.5 w-2.5">
      {#if $conductorStatus === 'connecting'}
        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
      {/if}
      <span class="relative inline-flex rounded-full h-2.5 w-2.5 {statusColor}"></span>
    </span>
    <span>
      {#if $conductorStatus === 'demo'}
        Demo Mode — showing simulated data (no Holochain conductor detected, retrying...)
      {:else if $conductorStatus === 'connecting'}
        Connecting to Holochain conductor...
      {:else if $conductorStatus === 'disconnected'}
        Disconnected from conductor
      {/if}
    </span>
    {#if $queueCount > 0}
      <span class="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-600 text-amber-50">
        {$queueCount} queued
      </span>
      <button
        on:click={handleSyncNow}
        disabled={syncing}
        class="ml-1 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500 hover:bg-amber-400 text-amber-950 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {syncing ? 'Syncing...' : 'Sync'}
      </button>
    {/if}
  </div>
{/if}

<!-- Navigation -->
<nav class="bg-gray-900 border-b border-gray-800" aria-label="Main navigation">
  <div class="container mx-auto px-4 py-2">
    <!-- Desktop nav -->
    <div class="hidden md:flex items-center gap-1 text-sm overflow-x-auto">
      {#each navGroups as group, i}
        {#if i > 0}
          <span class="text-gray-700 mx-1" aria-hidden="true">|</span>
        {/if}
        {#each group.items as item}
          <a href={item.href} class={linkClass(item.href)}>{item.label}</a>
        {/each}
      {/each}
      <span class="ml-auto" aria-hidden="true"></span>
      <span class="flex items-center gap-1.5 text-xs text-gray-400 pl-2 border-l border-gray-700" title={$connectionLabel}>
        <span class="relative flex h-2 w-2">
          {#if $connectionQuality === 'degraded'}
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full {$qualityColor} opacity-75"></span>
          {/if}
          <span class="relative inline-flex rounded-full h-2 w-2 {$qualityColor}"></span>
        </span>
        {$connectionLabel}
      </span>
      <span class="pl-2 border-l border-gray-700">
        <LocaleSelector />
      </span>
    </div>

    <!-- Mobile nav: hamburger + dropdown -->
    <div class="md:hidden flex items-center justify-between">
      <a href="/" class="text-white font-semibold text-sm">Mycelix</a>
      <button
        on:click={() => mobileMenuOpen = !mobileMenuOpen}
        class="p-2 rounded text-gray-300 hover:bg-gray-800 hover:text-white"
        aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
        aria-expanded={mobileMenuOpen}
      >
        {#if mobileMenuOpen}
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        {:else}
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        {/if}
      </button>
    </div>

    {#if mobileMenuOpen}
      <div class="md:hidden mt-2 pb-2 space-y-3">
        {#each navGroups as group}
          <div>
            <p class="text-xs text-gray-500 uppercase tracking-wider px-3 mb-1">{group.label}</p>
            <div class="grid grid-cols-2 gap-1">
              {#each group.items as item}
                <a href={item.href} class="{linkClass(item.href)} text-sm block">{item.label}</a>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</nav>

<Toast />
<slot />
