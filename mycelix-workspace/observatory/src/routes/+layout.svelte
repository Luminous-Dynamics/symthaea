<script lang="ts">
  import { conductorStatus } from '$lib/stores';

  $: statusLabel = {
    connecting: 'Connecting...',
    connected: 'Live',
    disconnected: 'Disconnected',
    demo: 'Demo Mode',
  }[$conductorStatus] ?? 'Unknown';

  $: statusColor = {
    connecting: 'bg-yellow-500',
    connected: 'bg-green-500',
    disconnected: 'bg-red-500',
    demo: 'bg-yellow-500',
  }[$conductorStatus] ?? 'bg-gray-500';

  $: statusTextColor = {
    connecting: 'text-yellow-400',
    connected: 'text-green-400',
    disconnected: 'text-red-400',
    demo: 'text-yellow-400',
  }[$conductorStatus] ?? 'text-gray-400';
</script>

<!-- Connection status bar — visible on every page -->
{#if $conductorStatus !== 'connected'}
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
  </div>
{/if}

<!-- Navigation — all existing Observatory routes -->
<nav class="bg-gray-900 border-b border-gray-800 px-4 py-2">
  <div class="container mx-auto flex items-center gap-1 overflow-x-auto text-sm">
    <a href="/" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Dashboard</a>
    <span class="text-gray-700">|</span>
    <a href="/tend" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">TEND</a>
    <a href="/food" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Food</a>
    <a href="/mutual-aid" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Mutual Aid</a>
    <a href="/emergency" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Emergency</a>
    <a href="/value-anchor" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Value Anchor</a>
    <span class="text-gray-700">|</span>
    <a href="/governance" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Governance</a>
    <a href="/network" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Network</a>
    <a href="/analytics" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Analytics</a>
    <a href="/attribution" class="px-3 py-1.5 rounded text-gray-300 hover:bg-gray-800 hover:text-white transition-colors whitespace-nowrap">Attribution</a>
  </div>
</nav>

<slot />
