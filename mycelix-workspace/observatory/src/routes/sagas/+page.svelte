<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
--><script lang="ts">
  import { onMount } from 'svelte';
  import { callZome, conductorStatus$ } from '$lib/conductor';

  // ============================================================================
  // Types
  // ============================================================================

  interface SagaStep {
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'compensated';
    started_at: number | null;
    completed_at: number | null;
    error: string | null;
  }

  interface Saga {
    id: string;
    name: string;
    status: 'Running' | 'Completed' | 'Failed' | 'Compensated';
    steps: SagaStep[];
    created_at: number;
  }

  // ============================================================================
  // State
  // ============================================================================

  let sagas: Saga[] = [];
  let loading = true;
  let error = '';
  let expandedId: string | null = null;

  // ============================================================================
  // Data fetching
  // ============================================================================

  onMount(async () => {
    if ($conductorStatus$ === 'disconnected') {
      loading = false;
      return;
    }
    try {
      const result = await callZome<Saga[]>({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'get_my_sagas',
        payload: null,
      });
      sagas = result ?? [];
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  });

  // ============================================================================
  // Helpers
  // ============================================================================

  function toggleExpand(id: string) {
    expandedId = expandedId === id ? null : id;
  }

  function getStatusBadge(status: string): string {
    switch (status) {
      case 'Running': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'Completed': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'Failed': return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'Compensated': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  }

  function getStepStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'text-green-400';
      case 'running': return 'text-blue-400';
      case 'failed': return 'text-red-400';
      case 'compensated': return 'text-yellow-400';
      case 'pending': return 'text-gray-500';
      default: return 'text-gray-400';
    }
  }

  function formatTimestamp(ts: number): string {
    return new Date(ts).toLocaleString();
  }
</script>

<svelte:head>
  <title>Sagas | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <!-- Page Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">&#x1F504;</span>
        <div>
          <h1 class="text-lg font-bold">Sagas Dashboard</h1>
          <p class="text-xs text-gray-400">Distributed Transaction Orchestration</p>
        </div>
      </div>
      <div class="text-right">
        <p class="text-xs text-gray-400">Total Sagas</p>
        <p class="text-lg font-bold">{sagas.length}</p>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Disconnected banner -->
    {#if $conductorStatus$ === 'disconnected' || $conductorStatus$ === 'demo'}
      <div class="bg-yellow-500/10 border border-yellow-500/50 rounded-lg p-4 mb-6 text-yellow-400 text-sm">
        Conductor is not connected. Saga data is unavailable until the Holochain conductor is running.
      </div>
    {/if}

    <!-- Loading -->
    {#if loading}
      <div class="flex items-center justify-center py-16">
        <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-400"></div>
        <span class="ml-3 text-gray-400">Loading sagas...</span>
      </div>

    <!-- Error -->
    {:else if error}
      <div class="bg-red-500/10 border border-red-500/50 rounded-lg p-4 text-red-400 text-sm">
        Failed to load sagas: {error}
      </div>

    <!-- Empty -->
    {:else if sagas.length === 0}
      <div class="text-center py-16 text-gray-500">
        <p class="text-lg">No sagas found</p>
        <p class="text-sm mt-1">Sagas will appear here when cross-zome transactions are initiated.</p>
      </div>

    <!-- Saga List -->
    {:else}
      <!-- Summary stats -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 class="text-gray-400 text-xs uppercase">Running</h3>
          <p class="text-2xl font-bold mt-1 text-blue-400">{sagas.filter(s => s.status === 'Running').length}</p>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 class="text-gray-400 text-xs uppercase">Completed</h3>
          <p class="text-2xl font-bold mt-1 text-green-400">{sagas.filter(s => s.status === 'Completed').length}</p>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 class="text-gray-400 text-xs uppercase">Failed</h3>
          <p class="text-2xl font-bold mt-1 text-red-400">{sagas.filter(s => s.status === 'Failed').length}</p>
        </div>
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 class="text-gray-400 text-xs uppercase">Compensated</h3>
          <p class="text-2xl font-bold mt-1 text-yellow-400">{sagas.filter(s => s.status === 'Compensated').length}</p>
        </div>
      </div>

      <!-- Saga cards -->
      <div class="space-y-4">
        {#each sagas as saga}
          <div class="bg-gray-800 rounded-lg border border-gray-700">
            <!-- Saga header (clickable) -->
            <button
              class="w-full p-4 flex justify-between items-center text-left hover:bg-gray-700/50 transition-colors rounded-lg"
              on:click={() => toggleExpand(saga.id)}
            >
              <div class="flex items-center gap-3">
                <span class={`text-xs px-2 py-0.5 rounded border ${getStatusBadge(saga.status)}`}>
                  {saga.status}
                </span>
                <div>
                  <p class="font-medium">{saga.name}</p>
                  <p class="text-xs text-gray-400 mt-0.5">{formatTimestamp(saga.created_at)}</p>
                </div>
              </div>
              <div class="flex items-center gap-4">
                <span class="text-sm text-gray-400">{saga.steps.length} steps</span>
                <span class="text-gray-500">{expandedId === saga.id ? '\u25B2' : '\u25BC'}</span>
              </div>
            </button>

            <!-- Expanded steps -->
            {#if expandedId === saga.id}
              <div class="border-t border-gray-700 p-4">
                <div class="space-y-2">
                  {#each saga.steps as step, i}
                    <div class="flex items-center gap-3 p-2 bg-gray-700/30 rounded">
                      <span class="text-gray-500 text-xs font-mono w-6 text-right">{i + 1}</span>
                      <span class={`text-sm font-medium ${getStepStatusColor(step.status)}`}>
                        {step.status.toUpperCase()}
                      </span>
                      <span class="text-sm">{step.name}</span>
                      {#if step.error}
                        <span class="text-xs text-red-400 ml-auto">{step.error}</span>
                      {/if}
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Sagas v0.1.0 &bull; Civic Bridge &bull; HDK 0.6.0</p>
    </footer>
  </main>
</div>
