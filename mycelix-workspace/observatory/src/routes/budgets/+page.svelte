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

  type CyclePhase = 'Proposal' | 'Deliberation' | 'Voting' | 'Execution' | 'Complete';

  interface ProjectAllocation {
    name: string;
    amount: number;
    votes_for: number;
    votes_against: number;
  }

  interface BudgetCycle {
    id: string;
    name: string;
    total_budget: number;
    phase: CyclePhase;
    proposal_deadline: number;
    voting_deadline: number;
    projects: ProjectAllocation[];
    created_at: number;
  }

  // ============================================================================
  // State
  // ============================================================================

  let cycles: BudgetCycle[] = [];
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
      const result = await callZome<BudgetCycle[]>({
        role_name: 'governance',
        zome_name: 'budgeting',
        fn_name: 'get_all_cycles',
        payload: null,
      });
      cycles = result ?? [];
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

  function getPhaseBadge(phase: CyclePhase): string {
    switch (phase) {
      case 'Proposal': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'Deliberation': return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
      case 'Voting': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'Execution': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      case 'Complete': return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  }

  function formatBudget(amount: number): string {
    if (amount >= 1_000_000) return `${(amount / 1_000_000).toFixed(1)}M SAP`;
    if (amount >= 1_000) return `${(amount / 1_000).toFixed(1)}K SAP`;
    return `${amount} SAP`;
  }

  function formatDate(ts: number): string {
    return new Date(ts).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  }

  function votePercentage(proj: ProjectAllocation): number {
    const total = proj.votes_for + proj.votes_against;
    if (total === 0) return 0;
    return (proj.votes_for / total) * 100;
  }
</script>

<svelte:head>
  <title>Budget Cycles | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <!-- Page Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">&#x1F4B0;</span>
        <div>
          <h1 class="text-lg font-bold">Budget Cycles</h1>
          <p class="text-xs text-gray-400">Participatory Budgeting</p>
        </div>
      </div>
      <div class="text-right">
        <p class="text-xs text-gray-400">Active Cycles</p>
        <p class="text-lg font-bold text-green-400">{cycles.filter(c => c.phase !== 'Complete').length}</p>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Disconnected banner -->
    {#if $conductorStatus$ === 'disconnected' || $conductorStatus$ === 'demo'}
      <div class="bg-yellow-500/10 border border-yellow-500/50 rounded-lg p-4 mb-6 text-yellow-400 text-sm">
        Conductor is not connected. Budget data is unavailable until the Holochain conductor is running.
      </div>
    {/if}

    <!-- Loading -->
    {#if loading}
      <div class="flex items-center justify-center py-16">
        <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-400"></div>
        <span class="ml-3 text-gray-400">Loading budget cycles...</span>
      </div>

    <!-- Error -->
    {:else if error}
      <div class="bg-red-500/10 border border-red-500/50 rounded-lg p-4 text-red-400 text-sm">
        Failed to load budget cycles: {error}
      </div>

    <!-- Empty -->
    {:else if cycles.length === 0}
      <div class="text-center py-16 text-gray-500">
        <p class="text-lg">No budget cycles</p>
        <p class="text-sm mt-1">Budget cycles will appear here when created through governance.</p>
      </div>

    <!-- Cycle List -->
    {:else}
      <!-- Summary stats -->
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        {#each ['Proposal', 'Deliberation', 'Voting', 'Execution', 'Complete'] as phase}
          <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 class="text-gray-400 text-xs uppercase">{phase}</h3>
            <p class="text-2xl font-bold mt-1">{cycles.filter(c => c.phase === phase).length}</p>
          </div>
        {/each}
      </div>

      <!-- Cycle cards -->
      <div class="space-y-4">
        {#each cycles as cycle}
          <div class="bg-gray-800 rounded-lg border border-gray-700">
            <!-- Cycle header (clickable) -->
            <button
              class="w-full p-4 flex justify-between items-center text-left hover:bg-gray-700/50 transition-colors rounded-lg"
              on:click={() => toggleExpand(cycle.id)}
            >
              <div class="flex items-center gap-3">
                <span class={`text-xs px-2 py-0.5 rounded border ${getPhaseBadge(cycle.phase)}`}>
                  {cycle.phase}
                </span>
                <div>
                  <p class="font-medium">{cycle.name}</p>
                  <div class="flex gap-4 text-xs text-gray-400 mt-0.5">
                    <span>Proposals by {formatDate(cycle.proposal_deadline)}</span>
                    <span>Voting by {formatDate(cycle.voting_deadline)}</span>
                  </div>
                </div>
              </div>
              <div class="flex items-center gap-4">
                <span class="text-sm font-semibold text-green-400">{formatBudget(cycle.total_budget)}</span>
                <span class="text-sm text-gray-400">{cycle.projects.length} projects</span>
                <span class="text-gray-500">{expandedId === cycle.id ? '\u25B2' : '\u25BC'}</span>
              </div>
            </button>

            <!-- Expanded projects -->
            {#if expandedId === cycle.id}
              <div class="border-t border-gray-700 p-4">
                {#if cycle.projects.length === 0}
                  <p class="text-sm text-gray-500">No projects submitted yet.</p>
                {:else}
                  <div class="space-y-3">
                    {#each cycle.projects as proj}
                      <div class="p-3 bg-gray-700/30 rounded">
                        <div class="flex justify-between items-center">
                          <span class="text-sm font-medium">{proj.name}</span>
                          <span class="text-sm text-green-400">{formatBudget(proj.amount)}</span>
                        </div>
                        <!-- Vote bar -->
                        <div class="mt-2">
                          <div class="flex justify-between text-xs mb-1">
                            <span class="text-green-400">For: {proj.votes_for}</span>
                            <span class="text-red-400">Against: {proj.votes_against}</span>
                          </div>
                          <div class="w-full bg-gray-600 rounded-full h-1.5 flex overflow-hidden">
                            <div class="bg-green-500 h-1.5" style="width: {votePercentage(proj)}%"></div>
                            <div class="bg-red-500 h-1.5" style="width: {100 - votePercentage(proj)}%"></div>
                          </div>
                        </div>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Budgets v0.1.0 &bull; Governance Budgeting &bull; HDK 0.6.0</p>
    </footer>
  </main>
</div>
