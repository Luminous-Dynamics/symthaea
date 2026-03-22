<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';

  interface HealthStatus {
    conductor_connected: boolean;
    conductor_url: string;
    version: string;
  }

  let health: HealthStatus | null = $state(null);
  let error: string | null = $state(null);

  onMount(async () => {
    try {
      health = await invoke<HealthStatus>('health_check');
    } catch (e) {
      error = String(e);
    }
  });
</script>

<main class="min-h-screen bg-gray-950 text-white p-8">
  <header class="mb-12">
    <h1 class="text-3xl font-bold tracking-tight">Terra Atlas Operator Console</h1>
    <p class="text-gray-400 mt-2">Decentralized energy project management</p>
  </header>

  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <!-- Connection Status -->
    <div class="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Conductor</h2>
      {#if health}
        <div class="flex items-center gap-2 mb-2">
          <div class="w-3 h-3 rounded-full {health.conductor_connected ? 'bg-green-500' : 'bg-yellow-500'}"></div>
          <span>{health.conductor_connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <p class="text-gray-500 text-sm">{health.conductor_url}</p>
        <p class="text-gray-600 text-xs mt-1">v{health.version}</p>
      {:else if error}
        <p class="text-red-400">{error}</p>
      {:else}
        <p class="text-gray-500">Connecting...</p>
      {/if}
    </div>

    <!-- Quick Links -->
    <div class="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Navigate</h2>
      <nav class="space-y-2">
        <a href="/projects" class="block text-blue-400 hover:text-blue-300">Projects</a>
        <a href="/investments" class="block text-blue-400 hover:text-blue-300">Investments</a>
        <a href="/consciousness" class="block text-blue-400 hover:text-blue-300">Consciousness Profile</a>
        <a href="/verify" class="block text-blue-400 hover:text-blue-300">Field Verification</a>
      </nav>
    </div>

    <!-- Consciousness Score -->
    <div class="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Your Profile</h2>
      <p class="text-gray-500">Consciousness tier and harmony alignment will appear here once connected to the Mycelix conductor.</p>
    </div>
  </div>
</main>
