<script lang="ts">
  import type { Writable } from 'svelte/store';
  import { timeAgo } from '$lib/freshness';

  export let lastUpdated: Writable<number>;
  export let loadError: Writable<string>;
  export let refreshing: Writable<boolean>;
  export let refresh: () => Promise<void>;

  let displayTime = 'never';
  let tickInterval: ReturnType<typeof setInterval> | null = null;

  // Update relative time every 10 seconds
  $: if ($lastUpdated) {
    displayTime = timeAgo($lastUpdated);
    if (!tickInterval) {
      tickInterval = setInterval(() => {
        displayTime = timeAgo($lastUpdated);
      }, 10_000);
    }
  }

  import { onDestroy } from 'svelte';
  onDestroy(() => { if (tickInterval) clearInterval(tickInterval); });
</script>

<div class="max-w-6xl mx-auto mb-4 flex items-center gap-3 text-xs text-gray-500" role="status" aria-live="polite">
  {#if $refreshing}
    <span class="inline-flex items-center gap-1.5">
      <span class="w-2 h-2 rounded-full bg-blue-400 animate-pulse" aria-hidden="true"></span>
      Refreshing...
    </span>
  {:else if $loadError}
    <span class="inline-flex items-center gap-1.5 text-red-400">
      <span class="w-2 h-2 rounded-full bg-red-400" aria-hidden="true"></span>
      {$loadError}
    </span>
    <button
      on:click={() => refresh()}
      class="text-gray-400 hover:text-white underline"
      aria-label="Retry data refresh"
    >Retry</button>
  {:else}
    <span class="inline-flex items-center gap-1.5">
      <span class="w-2 h-2 rounded-full bg-green-400" aria-hidden="true"></span>
      Updated {displayTime}
    </span>
  {/if}
  <button
    on:click={() => refresh()}
    disabled={$refreshing}
    class="ml-auto text-gray-500 hover:text-white transition-colors disabled:opacity-50"
    aria-label="Refresh data now"
    title="Refresh now"
  >
    <svg class="w-3.5 h-3.5 {$refreshing ? 'animate-spin' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  </button>
</div>
