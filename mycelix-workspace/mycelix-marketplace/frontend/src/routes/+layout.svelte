<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { initHolochainClient } from '$lib/holochain/client';
  import HolochainStatus from '$lib/components/HolochainStatus.svelte';
  import { holochain, favoritesCount } from '$lib/stores';
  import { getAvailableUrls, loadSelectedUrl, saveSelectedUrl } from '$lib/holochain/config';

  const shouldAutoconnect = import.meta.env?.VITE_HOLOCHAIN_AUTOCONNECT !== 'false';
  const urlOptions = getAvailableUrls();
  const defaultUrl = urlOptions[0]?.url || 'ws://localhost:8888';
  let selectedUrl = loadSelectedUrl(defaultUrl);

  // Initialize Holochain connection on app startup (configurable)
  onMount(async () => {
    if (!shouldAutoconnect) {
      console.info('Skipping Holochain autoconnect (VITE_HOLOCHAIN_AUTOCONNECT=false)');
      return;
    }

    try {
      await initHolochainClient(selectedUrl);
      console.log(`Holochain client initialized at ${selectedUrl}`);
    } catch (e) {
      console.error('Failed to connect to Holochain:', e);
    }
  });

  function handleUrlChange(url: string) {
    selectedUrl = url;
    saveSelectedUrl(url);
    initHolochainClient(url).catch((e) => console.error('Reconnect failed', e));
  }
</script>

<slot />
<HolochainStatus />

<div class="hc-chip" role="status" aria-live="polite" title={`Target: ${selectedUrl}`}>
  <span class={`dot ${$holochain.status}`}></span>
  <span class="label">
    {$holochain.status === 'connected'
      ? 'Holochain connected'
      : $holochain.status === 'connecting'
      ? 'Connecting...'
      : 'Offline'}
  </span>
  <select class="hc-select" bind:value={selectedUrl} on:change={(e) => handleUrlChange(e.currentTarget.value)}>
    {#each urlOptions as opt}
      <option value={opt.url}>{opt.label}</option>
    {/each}
  </select>
  <span class="fav-badge" title="Favorites saved locally">♥ {$favoritesCount}</span>
</div>

<style global>
  /* Any additional global styles */
  .hc-chip {
    position: fixed;
    top: 1rem;
    right: 1rem;
    background: rgba(26, 32, 44, 0.9);
    color: #f7fafc;
    padding: 0.4rem 0.75rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    z-index: 45;
    font-size: 0.9rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
  }

  .hc-chip .dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
    background: #e2e8f0;
  }

  .hc-chip .dot.connected {
    background: #48bb78;
  }

  .hc-chip .dot.connecting {
    background: #f6ad55;
    animation: pulse 1s infinite;
  }

  .hc-chip .dot.disconnected,
  .hc-chip .dot.error {
    background: #f56565;
  }

  .hc-select {
    background: rgba(255, 255, 255, 0.1);
    color: #f7fafc;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.5rem;
    padding: 0.25rem 0.4rem;
    font-size: 0.85rem;
  }

  .fav-badge {
    background: rgba(255, 255, 255, 0.12);
    color: #fbd38d;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 999px;
    padding: 0.25rem 0.5rem;
    font-weight: 700;
    font-size: 0.85rem;
  }

  @keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.3);
      opacity: 0.7;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  }
</style>
