<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
--><script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';

  interface AllocationDashboard {
    project_id: string;
    consciousness: { phi_score: number; harmony_alignment: number } | null;
    impact: {
      total_co2_avoided: number;
      total_jobs_created: number;
      total_trust_delta: number;
      peak_households: number;
      report_count: number;
      net_humanity_benefit: number;
    } | null;
    pending_pledges: number;
    matched_pledges: number;
    total_pledged: number;
    net_humanity_benefit: number;
    discovery_score: number;
  }

  interface ProjectSummary {
    id: string;
    name: string;
    project_type: string;
    phi_score: number | null;
  }

  let projects: ProjectSummary[] = $state([]);
  let selectedDashboard: AllocationDashboard | null = $state(null);
  let selectedName = $state('');
  let loading = $state(false);

  onMount(async () => {
    try {
      projects = await invoke<ProjectSummary[]>('get_projects');
    } catch (e) {
      console.warn('Demo mode:', e);
    }
  });

  async function loadDashboard(project: ProjectSummary) {
    selectedName = project.name;
    loading = true;
    try {
      selectedDashboard = await invoke<AllocationDashboard>('get_project_dashboard', {
        projectId: project.id,
      });
    } catch (e) {
      console.warn('Dashboard load error:', e);
      selectedDashboard = null;
    } finally {
      loading = false;
    }
  }
</script>

<main class="min-h-screen bg-gray-950 text-white p-8">
  <header class="mb-8">
    <a href="/" class="text-gray-400 hover:text-white text-sm mb-2 inline-block">&larr; Dashboard</a>
    <h1 class="text-2xl font-bold">Impact</h1>
    <p class="text-gray-400 text-sm mt-1">QOL metrics — the primary signal, not price</p>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Project selector -->
    <div class="space-y-2">
      <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Select Project</h2>
      {#each projects as project}
        <button
          onclick={() => loadDashboard(project)}
          class="w-full text-left rounded-lg p-3 border text-sm transition-all
            {selectedDashboard?.project_id === project.id
              ? 'bg-emerald-900/20 border-emerald-500/50'
              : 'bg-gray-900 border-gray-800 hover:border-gray-700'}"
        >
          <span>{project.name}</span>
          <span class="text-gray-500 text-xs block">{project.project_type}</span>
        </button>
      {/each}
    </div>

    <!-- Impact Dashboard -->
    <div class="lg:col-span-2">
      {#if loading}
        <div class="bg-gray-900 rounded-xl p-8 border border-gray-800 animate-pulse">
          <div class="h-6 bg-gray-800 rounded w-1/3 mb-4"></div>
          <div class="grid grid-cols-2 gap-4">
            <div class="h-20 bg-gray-800 rounded"></div>
            <div class="h-20 bg-gray-800 rounded"></div>
            <div class="h-20 bg-gray-800 rounded"></div>
            <div class="h-20 bg-gray-800 rounded"></div>
          </div>
        </div>
      {:else if selectedDashboard}
        <div class="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 class="text-lg font-semibold mb-1">{selectedName}</h3>

          <!-- Trust Score -->
          <div class="flex items-center gap-3 mb-6">
            {#if selectedDashboard.consciousness}
              <span class="font-mono text-xl text-violet-400">
                &#934; {(selectedDashboard.consciousness.phi_score * 100).toFixed(0)}
              </span>
              <span class="font-mono text-sm text-blue-400">
                &#8801; {(selectedDashboard.consciousness.harmony_alignment * 100).toFixed(0)}
              </span>
            {:else}
              <span class="text-gray-500 text-sm">No trust data yet</span>
            {/if}
            <span class="text-xs text-gray-500 ml-auto">
              NHB: <span class="font-mono text-emerald-400">{selectedDashboard.net_humanity_benefit.toFixed(3)}</span>
            </span>
          </div>

          <!-- QOL Metrics -->
          <div class="grid grid-cols-2 gap-4 mb-6">
            <div class="bg-gray-800/60 rounded-lg p-4">
              <span class="text-xs text-gray-400 block mb-1">CO2 Avoided</span>
              <span class="font-mono text-lg text-green-400">
                {selectedDashboard.impact?.total_co2_avoided?.toLocaleString() ?? '0'} t
              </span>
            </div>
            <div class="bg-gray-800/60 rounded-lg p-4">
              <span class="text-xs text-gray-400 block mb-1">Jobs Created</span>
              <span class="font-mono text-lg text-blue-400">
                {selectedDashboard.impact?.total_jobs_created ?? 0}
              </span>
            </div>
            <div class="bg-gray-800/60 rounded-lg p-4">
              <span class="text-xs text-gray-400 block mb-1">Trust Delta</span>
              <span class="font-mono text-lg {(selectedDashboard.impact?.total_trust_delta ?? 0) >= 0 ? 'text-rose-400' : 'text-orange-400'}">
                {((selectedDashboard.impact?.total_trust_delta ?? 0) >= 0 ? '+' : '')}{((selectedDashboard.impact?.total_trust_delta ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div class="bg-gray-800/60 rounded-lg p-4">
              <span class="text-xs text-gray-400 block mb-1">Households</span>
              <span class="font-mono text-lg text-cyan-400">
                {selectedDashboard.impact?.peak_households ?? 0}
              </span>
            </div>
          </div>

          <!-- Allocation Status -->
          <div class="border-t border-gray-800 pt-4">
            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Allocation Status</h4>
            <div class="flex items-center gap-6 text-sm">
              <div>
                <span class="text-gray-400">Pledged:</span>
                <span class="font-mono ml-1">{selectedDashboard.total_pledged} TEND</span>
              </div>
              <div>
                <span class="text-gray-400">Pending:</span>
                <span class="font-mono ml-1 text-amber-400">{selectedDashboard.pending_pledges}</span>
              </div>
              <div>
                <span class="text-gray-400">Matched:</span>
                <span class="font-mono ml-1 text-emerald-400">{selectedDashboard.matched_pledges}</span>
              </div>
            </div>
          </div>

          <!-- NHB Bar -->
          <div class="mt-4">
            <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                class="h-full bg-gradient-to-r from-emerald-500 to-green-400 rounded-full transition-all duration-500"
                style="width: {Math.min(Math.max(selectedDashboard.net_humanity_benefit * 100, 0), 100)}%"
              ></div>
            </div>
            <p class="text-xs text-gray-600 mt-1 text-center">Net Humanity Benefit</p>
          </div>
        </div>

        <div class="mt-4 bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
          <p class="text-xs text-gray-500">
            All metrics sourced from Holochain DHT. No centralized database. Impact reports are verified on-chain.
          </p>
        </div>
      {:else}
        <div class="bg-gray-900 rounded-xl p-8 border border-gray-800 text-center">
          <p class="text-gray-500">Select a project to view its impact dashboard.</p>
          <p class="text-gray-600 text-xs mt-2">Data loaded directly from the Holochain conductor — no servers.</p>
        </div>
      {/if}
    </div>
  </div>
</main>
