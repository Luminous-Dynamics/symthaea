<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
--><script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';

  interface HealthStatus {
    conductor: {
      connected: boolean;
      url: string;
      last_ping_ms: number | null;
      agent_key: string | null;
    };
    version: string;
  }

  interface ProjectSummary {
    id: string;
    name: string;
    project_type: string;
    capacity_mw: number;
    status: string;
    phi_score: number | null;
    harmony_alignment: number | null;
  }

  interface AllocationDashboard {
    project_id: string;
    pending_pledges: number;
    matched_pledges: number;
    total_pledged: number;
    net_humanity_benefit: number;
    discovery_score: number;
  }

  let health: HealthStatus | null = $state(null);
  let projects: ProjectSummary[] = $state([]);
  let error: string | null = $state(null);
  let connecting = $state(false);

  onMount(async () => {
    try {
      health = await invoke<HealthStatus>('health_check');
      if (health?.conductor.connected) {
        await loadProjects();
      }
    } catch (e) {
      error = String(e);
    }
  });

  async function connectConductor() {
    connecting = true;
    try {
      await invoke('connect_conductor', { url: null });
      health = await invoke<HealthStatus>('health_check');
      if (health?.conductor.connected) {
        await loadProjects();
      }
    } catch (e) {
      error = String(e);
    } finally {
      connecting = false;
    }
  }

  async function loadProjects() {
    try {
      projects = await invoke<ProjectSummary[]>('get_projects');
    } catch (e) {
      // Demo data returned by default when conductor not fully wired
      console.warn('Using demo projects:', e);
    }
  }

  function trustColor(score: number | null): string {
    if (score === null) return 'text-gray-500';
    if (score >= 0.8) return 'text-violet-400';
    if (score >= 0.6) return 'text-blue-400';
    if (score >= 0.4) return 'text-cyan-400';
    if (score >= 0.2) return 'text-amber-400';
    return 'text-gray-400';
  }

  function trustLabel(score: number | null): string {
    if (score === null) return 'Unscored';
    if (score >= 0.7) return 'Aligned';
    if (score >= 0.4) return 'Emerging';
    return 'Nascent';
  }
</script>

<main class="min-h-screen bg-gray-950 text-white p-8">
  <header class="mb-8">
    <h1 class="text-3xl font-bold tracking-tight">Trust-Weighted Allocation Network</h1>
    <p class="text-gray-400 mt-1">Decentralized resource allocation via Holochain — no centralized servers</p>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
    <!-- Conductor Status -->
    <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
      <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Conductor</h2>
      {#if health}
        <div class="flex items-center gap-2 mb-2">
          <div class="w-2.5 h-2.5 rounded-full {health.conductor.connected ? 'bg-emerald-500 shadow-lg shadow-emerald-500/30' : 'bg-yellow-500'}"></div>
          <span class="text-sm">{health.conductor.connected ? 'Connected' : 'Disconnected'}</span>
        </div>
        <p class="text-gray-600 text-xs">{health.conductor.url}</p>
        {#if health.conductor.last_ping_ms}
          <p class="text-gray-600 text-xs">{health.conductor.last_ping_ms}ms latency</p>
        {/if}
        {#if !health.conductor.connected}
          <button onclick={connectConductor} disabled={connecting}
            class="mt-3 w-full px-3 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 rounded-lg disabled:opacity-50">
            {connecting ? 'Connecting...' : 'Connect'}
          </button>
        {/if}
      {:else if error}
        <p class="text-red-400 text-sm">{error}</p>
      {:else}
        <div class="animate-pulse h-4 bg-gray-800 rounded w-2/3"></div>
      {/if}
    </div>

    <!-- Navigate -->
    <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
      <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Navigate</h2>
      <nav class="space-y-1.5 text-sm">
        <a href="/projects" class="block text-blue-400 hover:text-blue-300 transition">Projects</a>
        <a href="/allocations" class="block text-violet-400 hover:text-violet-300 transition">Allocations</a>
        <a href="/impact" class="block text-emerald-400 hover:text-emerald-300 transition">Impact</a>
        <a href="/verify" class="block text-cyan-400 hover:text-cyan-300 transition">Field Verify</a>
      </nav>
    </div>

    <!-- Trust Profile -->
    <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
      <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Your Trust Profile</h2>
      {#if health?.conductor.connected}
        <p class="text-gray-400 text-sm">Trust profile loads from your Holochain source chain — no server involved.</p>
      {:else}
        <p class="text-gray-500 text-sm">Connect to conductor to view your trust tier and harmony alignment.</p>
      {/if}
    </div>

    <!-- Network Stats -->
    <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
      <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Network</h2>
      <div class="space-y-2 text-sm">
        <div class="flex justify-between">
          <span class="text-gray-400">Projects</span>
          <span class="font-mono">{projects.length}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-400">Architecture</span>
          <span class="text-emerald-400 text-xs">Agent-centric</span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-400">Server</span>
          <span class="text-emerald-400 text-xs">None (P2P)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Harmony Leaderboard -->
  <div class="bg-gray-900 rounded-xl p-6 border border-gray-800 mb-8">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-sm font-semibold text-gray-200">Harmony Leaderboard</h2>
      <span class="text-xs text-gray-500">Ranked by trust score + impact</span>
    </div>

    {#if projects.length === 0}
      <p class="text-gray-500 text-sm">
        {health?.conductor.connected
          ? 'No projects registered on DHT yet.'
          : 'Connect to conductor to view projects.'}
      </p>
    {:else}
      <div class="space-y-2">
        {#each projects.sort((a, b) => (b.phi_score ?? 0) - (a.phi_score ?? 0)) as project, i}
          <div class="flex items-center gap-3 rounded-lg bg-gray-800/40 hover:bg-gray-800/70 transition-colors p-3 cursor-pointer">
            <span class="font-mono font-bold text-sm w-6 text-center {i === 0 ? 'text-amber-400' : i === 1 ? 'text-gray-300' : i === 2 ? 'text-orange-400' : 'text-gray-500'}">
              {i + 1}
            </span>
            <div class="flex-1 min-w-0">
              <span class="text-sm text-gray-200 truncate block">{project.name}</span>
              <span class="text-xs text-gray-500">{project.project_type} &middot; {project.capacity_mw} MW &middot; {project.status}</span>
            </div>
            <div class="flex items-center gap-2">
              {#if project.phi_score !== null}
                <span class="font-mono text-sm {trustColor(project.phi_score)}" title="Trust Score (Phi)">
                  &#934; {(project.phi_score * 100).toFixed(0)}
                </span>
              {/if}
              {#if project.harmony_alignment !== null}
                <span class="font-mono text-xs {trustColor(project.harmony_alignment)}" title="Harmony Alignment">
                  &#8801; {(project.harmony_alignment * 100).toFixed(0)}
                </span>
              {/if}
              <span class="text-xs {trustColor(project.phi_score)} bg-gray-800 px-2 py-0.5 rounded">
                {trustLabel(project.phi_score)}
              </span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <footer class="text-center text-gray-600 text-xs mt-12">
    <p>All data lives on your Holochain source chain. No centralized server. No data custody.</p>
    <p class="mt-1">Trust-Weighted Allocation Network v{health?.version ?? '0.1.0'}</p>
  </footer>
</main>
