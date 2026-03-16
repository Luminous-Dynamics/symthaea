<script lang="ts">
  import { timeAgo } from '$lib/freshness';
  import type { Writable } from 'svelte/store';

  export let lastUpdated: Writable<number>;
  export let loadError: Writable<string>;
  export let refreshing: Writable<boolean>;
  export let refresh: () => Promise<void>;

  let agoText = '';
  let agoTimer: ReturnType<typeof setInterval>;

  $: if ($lastUpdated) {
    agoText = timeAgo($lastUpdated);
  }

  import { onMount, onDestroy } from 'svelte';

  onMount(() => {
    agoTimer = setInterval(() => {
      if ($lastUpdated) agoText = timeAgo($lastUpdated);
    }, 10_000);
  });

  onDestroy(() => {
    if (agoTimer) clearInterval(agoTimer);
  });
</script>

{#if $loadError}
  <div class="bg-orange-900/60 border border-orange-700 text-orange-200 px-4 py-2 rounded-lg text-sm flex items-center justify-between mb-4">
    <span>Loaded offline — {$loadError}</span>
    <button on:click={refresh} disabled={$refreshing}
      class="ml-3 px-2 py-1 bg-orange-700 hover:bg-orange-600 disabled:opacity-50 rounded text-xs transition-colors">
      {$refreshing ? 'Retrying...' : 'Retry'}
    </button>
  </div>
{/if}

{#if $lastUpdated > 0}
  <div class="flex items-center justify-end gap-2 text-xs text-gray-500 mb-2">
    <span>Updated {agoText}</span>
    <button on:click={refresh} disabled={$refreshing}
      class="px-2 py-0.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded transition-colors"
      aria-label="Refresh data">
      {$refreshing ? '...' : 'Refresh'}
    </button>
  </div>
{/if}
